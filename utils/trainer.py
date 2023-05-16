import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from .metric import Metric
from .loss import LabelSmoothingLoss


class Trainer:
    def __init__(self, model, config, writer_dict, logger, rank=-1):
        self.model = model
        self.config = config
        self.writer_dict = writer_dict
        self.logger = logger
        self.rank = rank
        self.tgt_pad_idx = self.model.module.tgt_embedding.padding_idx
        self.criterion_bond_rc = nn.BCELoss(reduction='sum')
        self.criterion_atom_rc = nn.BCELoss(reduction='sum')
        self.criterion_context_align = LabelSmoothingLoss(reduction='sum', smoothing=0.5)
        self.criterion_tokens = LabelSmoothingLoss(ignore_index=self.tgt_pad_idx,
                                                   reduction='sum', apply_logsoftmax=False)
        self.cur_epoch = 0
        self.end_epoch = config.TRAIN.EPOCH
        self.cur_iter = 0
        self.best_accuracy = 0.0
        self.test_accuracy = 0.0
        self.early_stop = 0

    def train(self, train_loader, optimizer):
        self.model.train()
        metric = Metric(self.tgt_pad_idx)
        st_time = time.time()
        for batch in tqdm(train_loader, desc='(Train)', leave=False):
            torch.cuda.empty_cache()
            src, tgt, gt_context_alignment, gt_nonreactive_mask, src_graph = batch
            bond, _ = src_graph
            src, tgt, gt_context_alignment, gt_nonreactive_mask, bond = src.cuda(), tgt.cuda(), \
                                                                  gt_context_alignment.cuda(), \
                                                                  gt_nonreactive_mask.cuda(),\
                                                                  bond.cuda()
            p = np.random.rand()
            my_context = self.model.no_sync if self.rank != -1 and (
                    self.cur_iter + 1) % self.config.TRAIN.ACCUMULATION_STEPS != 0 else nullcontext
            with my_context():
                if p > self.anneal_prob(self.cur_iter):
                    generative_scores, atom_rc_scores, bond_rc_scores, context_scores = \
                        self.model(src, tgt, bond, gt_nonreactive_mask)
                else:
                    generative_scores, atom_rc_scores, bond_rc_scores, context_scores = \
                        self.model(src, tgt, bond, None)
                # language modeling loss
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                gt_token_label = tgt[1:].view(-1)

                # reaction center loss
                reaction_center_attn = ~gt_nonreactive_mask
                pred_atom_rc_logit = atom_rc_scores.view(-1)
                gt_atom_rc_label = reaction_center_attn.view(-1)

                if bond_rc_scores is not None:
                    pair_indices = torch.where(bond.sum(-1) > 0)
                    pred_bond_rc_prob = bond_rc_scores.view(-1)
                    gt_bond_rc_label = (reaction_center_attn[[pair_indices[1], pair_indices[0]]] & reaction_center_attn[
                        [pair_indices[2], pair_indices[0]]])
                    loss_bond_rc = self.criterion_bond_rc(pred_bond_rc_prob, gt_bond_rc_label.float())
                else:
                    loss_bond_rc = torch.zeros(1).to(src.device)

                # loss for context alignment
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                # add all loss
                loss_token = self.criterion_tokens(pred_token_logit, gt_token_label)
                loss_atom_rc = self.criterion_atom_rc(pred_atom_rc_logit, gt_atom_rc_label.float())
                loss_context_align = 0
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])
                loss_context_align += self.criterion_context_align(pred_context_align_logit, gt_context_align_label)
                loss = loss_token + loss_atom_rc + loss_bond_rc + loss_context_align
                loss.backward()
            if ((self.cur_iter + 1) % self.config.TRAIN.ACCUMULATION_STEPS) == 0:
                optimizer.step_and_update_lr()
                optimizer.zero_grad()
            self.cur_iter += 1
            metric.update(generative_scores.transpose(0, 1).contiguous().view(-1, generative_scores.size(2)),
                          (tgt.transpose(0, 1))[:, 1:].contiguous().view(-1),
                          loss.item() * self.config.TRAIN.ACCUMULATION_STEPS)
        self.cur_epoch += 1
        loss_per_word, top1_accuracy, topk_accuracy = metric.compute()
        top1_accuracy = top1_accuracy * 100
        topk_accuracy = topk_accuracy * 100
        msg = 'Epoch: [{}/{}], ppl: {:8.5f}, accuracy: {:3.3f} %,accuracy top{}: {:3.3f} %, lr: {:8.5f}, ' \
              'elapse: {:3.3f} min'.format(
            self.cur_epoch, self.end_epoch, math.exp(min(loss_per_word, 100)), top1_accuracy,
            self.config.TEST.TOPK, topk_accuracy, optimizer._optimizer.param_groups[0]['lr'],
            (time.time() - st_time) / 60)
        self.logger.info(msg)
        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['train_global_steps']
        if self.rank < 1:
            writer.add_scalar('train_accuracy', top1_accuracy, global_steps)
            writer.add_scalar('train_loss', loss_per_word, global_steps)
            writer.add_scalar('train_ppl', math.exp(min(loss_per_word, 100)), global_steps)
            writer.add_scalar('learning rate', optimizer._optimizer.param_groups[0]['lr'], global_steps)
            self.writer_dict['train_global_steps'] = global_steps + 1

    @staticmethod
    def anneal_prob(step, k=2, total=150000):
        step = np.clip(step, 0, total)
        min_, max_ = 1, np.exp(k * 1)
        return (np.exp(k * step / total) - min_) / (max_ - min_)

    def val(self, val_loader):
        self.model.eval()
        metric = Metric(self.tgt_pad_idx)
        st_time = time.time()
        pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
        pred_arc_list, gt_arc_list = [], []
        pred_brc_list, gt_brc_list = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='(val)', leave=False):
                torch.cuda.empty_cache()
                src, tgt, gt_context_alignment, gt_nonreactive_mask, src_graph = batch
                bond, _ = src_graph
                src, tgt, gt_context_alignment, gt_nonreactive_mask = src.cuda(), tgt.cuda(), \
                                                                      gt_context_alignment.cuda(), \
                                                                      gt_nonreactive_mask.cuda()

                generative_scores, atom_rc_scores, bond_rc_scores, context_scores = \
                    self.model(src, tgt, bond, None)
                context_alignment = F.softmax(context_scores[-1], dim=-1)
                # language modeling loss
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                gt_token_label = tgt[1:].view(-1)

                # reaction center loss
                reaction_center_attn = ~gt_nonreactive_mask
                pred_atom_rc_logit = atom_rc_scores.view(-1)
                gt_atom_rc_label = reaction_center_attn.view(-1)

                if bond_rc_scores is not None:
                    pair_indices = torch.where(bond.sum(-1) > 0)
                    pred_bond_rc_prob = bond_rc_scores.view(-1)
                    gt_bond_rc_label = (
                            reaction_center_attn[[pair_indices[1], pair_indices[0]]] & reaction_center_attn[
                        [pair_indices[2], pair_indices[0]]])
                    loss_bond_rc = self.criterion_bond_rc(pred_bond_rc_prob, gt_bond_rc_label.float())
                else:
                    loss_bond_rc = torch.zeros(1).to(src.device)

                # loss for context alignment
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                # add all loss
                loss_token = self.criterion_tokens(pred_token_logit, gt_token_label)
                loss_atom_rc = self.criterion_atom_rc(pred_atom_rc_logit, gt_atom_rc_label.float())
                loss_context_align = 0
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])
                loss_context_align += self.criterion_context_align(pred_context_align_logit, gt_context_align_label)
                loss = loss_token + loss_atom_rc + loss_bond_rc + loss_context_align
                # Atom-level reaction center accuracy:
                pred_arc = (atom_rc_scores.squeeze(2) > 0.5).bool()
                pred_arc_list += list(~pred_arc.view(-1).cpu().numpy())
                gt_arc_list += list(gt_nonreactive_mask.view(-1).cpu().numpy())
                gt_brc_list += list(gt_bond_rc_label.view(-1).cpu().numpy())

                # Bond-level reaction center accuracy:
                if bond_rc_scores is not None:
                    pred_brc = (bond_rc_scores > 0.5).bool()
                    pred_brc_list += list(pred_brc.view(-1).cpu().numpy())

                metric.update(generative_scores.transpose(0, 1).contiguous().view(-1, generative_scores.size(2)),
                              (tgt.transpose(0, 1))[:, 1:].contiguous().view(-1),
                              loss.item() * self.config.TRAIN.ACCUMULATION_STEPS)

        loss_per_word, top1_accuracy, topk_accuracy = metric.compute()
        top1_accuracy = top1_accuracy * 100
        topk_accuracy = topk_accuracy * 100
        msg = 'Validating result:, ppl: {:8.5f}, accuracy: {:3.3f} %,accuracy top{}: {:3.3f} %, ' \
              'elapse: {:3.3f} min'.format(
            math.exp(min(loss_per_word, 100)), top1_accuracy,
            self.config.TEST.TOPK, topk_accuracy, (time.time() - st_time) / 60)
        self.logger.info(msg)
        a_ac = 0
        b_ac = 0
        if bond_rc_scores is not None:
            a_ac = np.mean(np.array(pred_arc_list) == np.array(gt_arc_list))
            b_ac = np.mean(np.array(pred_brc_list) == np.array(gt_brc_list))
        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['valid_global_steps']
        if self.rank < 1:
            writer.add_scalar('valid_loss', loss_per_word, global_steps)
            writer.add_scalar('valid_accuracy', top1_accuracy, global_steps)
            writer.add_scalar('valid atom_center acc', a_ac, global_steps)
            writer.add_scalar('valid bond_center acc', b_ac, global_steps)
            writer.add_scalar('valid_ppl', math.exp(min(loss_per_word, 100)), global_steps)
            self.writer_dict['valid_global_steps'] = global_steps + self.config.TEST.EVAL_STEPS

    def test(self, test_loader):
        self.model.eval()
        metric = Metric(self.tgt_pad_idx)
        st_time = time.time()
        pred_token_list, gt_token_list, pred_infer_list, gt_infer_list = [], [], [], []
        pred_arc_list, gt_arc_list = [], []
        pred_brc_list, gt_brc_list = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='(Test)', leave=False):
                torch.cuda.empty_cache()
                src, tgt, gt_context_alignment, gt_nonreactive_mask, src_graph = batch
                bond, _ = src_graph
                src, tgt, gt_context_alignment, gt_nonreactive_mask = src.cuda(), tgt.cuda(), \
                                                                      gt_context_alignment.cuda(), \
                                                                      gt_nonreactive_mask.cuda()

                generative_scores, atom_rc_scores, bond_rc_scores, context_scores = \
                    self.model(src, tgt, bond, None)
                context_alignment = F.softmax(context_scores[-1], dim=-1)
                # language modeling loss
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                gt_token_label = tgt[1:].view(-1)

                # reaction center loss
                reaction_center_attn = ~gt_nonreactive_mask
                pred_atom_rc_logit = atom_rc_scores.view(-1)
                gt_atom_rc_label = reaction_center_attn.view(-1)

                if bond_rc_scores is not None:
                    pair_indices = torch.where(bond.sum(-1) > 0)
                    pred_bond_rc_prob = bond_rc_scores.view(-1)
                    gt_bond_rc_label = (
                            reaction_center_attn[[pair_indices[1], pair_indices[0]]] & reaction_center_attn[
                        [pair_indices[2], pair_indices[0]]])
                    loss_bond_rc = self.criterion_bond_rc(pred_bond_rc_prob, gt_bond_rc_label.float())
                else:
                    loss_bond_rc = torch.zeros(1).to(src.device)

                # loss for context alignment
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])

                # add all loss
                loss_token = self.criterion_tokens(pred_token_logit, gt_token_label)
                loss_atom_rc = self.criterion_atom_rc(pred_atom_rc_logit, gt_atom_rc_label.float())
                loss_context_align = 0
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])
                loss_context_align += self.criterion_context_align(pred_context_align_logit, gt_context_align_label)
                loss = loss_token + loss_atom_rc + loss_bond_rc + loss_context_align
                # Atom-level reaction center accuracy:
                pred_arc = (atom_rc_scores.squeeze(2) > 0.5).bool()
                pred_arc_list += list(~pred_arc.view(-1).cpu().numpy())
                gt_arc_list += list(gt_nonreactive_mask.view(-1).cpu().numpy())
                gt_brc_list += list(gt_bond_rc_label.view(-1).cpu().numpy())

                # Bond-level reaction center accuracy:
                if bond_rc_scores is not None:
                    pred_brc = (bond_rc_scores > 0.5).bool()
                    pred_brc_list += list(pred_brc.view(-1).cpu().numpy())

                metric.update(generative_scores.transpose(0, 1).contiguous().view(-1, generative_scores.size(2)),
                              (tgt.transpose(0, 1))[:, 1:].contiguous().view(-1),
                              loss.item() * self.config.TRAIN.ACCUMULATION_STEPS)

        loss_per_word, top1_accuracy, topk_accuracy = metric.compute()
        top1_accuracy = top1_accuracy * 100
        topk_accuracy = topk_accuracy * 100
        msg = 'Validating result:, ppl: {:8.5f}, accuracy: {:3.3f} %,accuracy top{}: {:3.3f} %, ' \
              'elapse: {:3.3f} min'.format(
            math.exp(min(loss_per_word, 100)), top1_accuracy,
            self.config.TEST.TOPK, topk_accuracy, (time.time() - st_time) / 60)
        self.logger.info(msg)
        a_ac = 0
        b_ac = 0
        if bond_rc_scores is not None:
            a_ac = np.mean(np.array(pred_arc_list) == np.array(gt_arc_list))
            b_ac = np.mean(np.array(pred_brc_list) == np.array(gt_brc_list))
        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['test_global_steps']
        if self.rank < 1:
            writer.add_scalar('test_loss', loss_per_word, global_steps)
            writer.add_scalar('test_accuracy', top1_accuracy, global_steps)
            writer.add_scalar('test atom_center acc', a_ac, global_steps)
            writer.add_scalar('test bond_center acc', b_ac, global_steps)
            writer.add_scalar('test_ppl', math.exp(min(loss_per_word, 100)), global_steps)
            self.writer_dict['test_global_steps'] = global_steps + self.config.TEST.EVAL_STEPS
        self.test_accuracy = top1_accuracy

    def save_models(self, output_dir):
        if self.rank < 1:
            checkpoint = {'epoch': self.cur_epoch, 'settings': self.config, 'model': self.model.state_dict()}
            if self.test_accuracy > self.best_accuracy:
                self.best_accuracy = self.test_accuracy
                model_path = os.path.join(output_dir, 'model.chkpt')
                torch.save(checkpoint, model_path)
                self.logger.info('The best checkpoint file has been updated.')
                self.early_stop = 0
            if self.config.TRAIN.SAVE_MODEL == 'all':
                model_path = os.path.join(output_dir, 'epoch:{}_model.chkpt'.format(self.cur_epoch))
                torch.save(checkpoint, model_path)
                self.logger.info('The checkpoint file has been saved.')
            self.logger.info('Best Accuracy: {:3.3f} %'.format(self.best_accuracy))
            self.early_stop += 1
        return self.early_stop >= 20