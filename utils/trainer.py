import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch
from contextlib import nullcontext

from .metric import Metric
from .loss import LabelSmoothingLoss, RDrop_Loss


class Trainer:
    def __init__(self, model, config, writer_dict, logger, rank=-1):
        self.model = model
        self.config = config
        self.writer_dict = writer_dict
        self.logger = logger
        self.rank = rank
        self.tgt_pad_idx = self.model.module.tgt_embedding.padding_idx
        self.criterion_context_align = LabelSmoothingLoss(reduction='sum', smoothing=0.5)
        self.criterion_tokens = RDrop_Loss(ignore_index=self.tgt_pad_idx, reduction='sum')
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
            src, tgt, gt_context_alignment, src_graph, src_threed, src_atoms = batch
            bond, _ = src_graph
            dist, _ = src_threed
            atoms_coord, atoms_token, atoms_index, batch_index = src_atoms
            src, tgt, gt_context_alignment = src.cuda(), tgt.cuda(), gt_context_alignment.cuda()
            bond = bond.cuda()
            dist = dist.cuda()
            atoms_coord, atoms_token, atoms_index, batch_index = \
                    atoms_coord.cuda(), atoms_token.cuda(), atoms_index.cuda(), batch_index.cuda()
            p = np.random.rand()
            my_context = self.model.no_sync if self.rank != -1 and (
                    self.cur_iter + 1) % self.config.TRAIN.ACCUMULATION_STEPS != 0 else nullcontext
            with my_context():
                generative_scores, context_scores = \
                    self.model(src, tgt, bond, dist, \
                                atoms_coord, atoms_token, atoms_index, batch_index)
                generative_scores_1, _ = \
                    self.model(src, tgt, bond, dist, \
                                atoms_coord, atoms_token, atoms_index, batch_index)
                # language modeling loss
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                pred_token_logit_1 = generative_scores_1.view(-1, generative_scores_1.size(2))
                gt_token_label = tgt[1:].view(-1)
                loss_token = self.criterion_tokens(pred_token_logit, pred_token_logit_1, gt_token_label)

                # loss for context alignment
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])
                loss_context_align = self.criterion_context_align(pred_context_align_logit, gt_context_align_label)

                # add all loss
                loss = loss_token + loss_context_align
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

    def val(self, val_loader):
        self.model.eval()
        metric = Metric(self.tgt_pad_idx)
        st_time = time.time()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='(val)', leave=False):
                src, tgt, gt_context_alignment, src_graph, src_threed, src_atoms = batch
                bond, _ = src_graph
                dist, _ = src_threed
                atoms_coord, atoms_token, atoms_index, batch_index = src_atoms
                src, tgt, gt_context_alignment = src.cuda(), tgt.cuda(), \
                                                gt_context_alignment.cuda()
                bond = bond.cuda()
                dist = dist.cuda()
                atoms_coord, atoms_token, atoms_index, batch_index = \
                    atoms_coord.cuda(), atoms_token.cuda(), atoms_index.cuda(), batch_index.cuda()
                generative_scores, context_scores = \
                    self.model(src, tgt, bond, dist, \
                               atoms_coord, atoms_token, atoms_index, batch_index)
                generative_scores_1, _ = \
                    self.model(src, tgt, bond, dist, \
                               atoms_coord, atoms_token, atoms_index, batch_index)
                # language modeling loss
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                pred_token_logit_1 = generative_scores_1.view(-1, generative_scores_1.size(2))
                gt_token_label = tgt[1:].view(-1)
                loss_token = self.criterion_tokens(pred_token_logit, pred_token_logit_1, gt_token_label)

                # loss for context alignment
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])
                loss_context_align = self.criterion_context_align(pred_context_align_logit, gt_context_align_label)
                
                # add all loss
                loss = loss_token + loss_context_align

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
        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['valid_global_steps']
        if self.rank < 1:
            writer.add_scalar('valid_loss', loss_per_word, global_steps)
            writer.add_scalar('valid_accuracy', top1_accuracy, global_steps)
            writer.add_scalar('valid_ppl', math.exp(min(loss_per_word, 100)), global_steps)
            self.writer_dict['valid_global_steps'] = global_steps + self.config.TEST.EVAL_STEPS

    def test(self, test_loader):
        self.model.eval()
        metric = Metric(self.tgt_pad_idx)
        st_time = time.time()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='(Test)', leave=False):
                src, tgt, gt_context_alignment, src_graph, src_threed, src_atoms = batch
                bond, _ = src_graph
                dist, _ = src_threed
                atoms_coord, atoms_token, atoms_index, batch_index = src_atoms
                src, tgt, gt_context_alignment = src.cuda(), tgt.cuda(), \
                                                gt_context_alignment.cuda()
                bond = bond.cuda()
                dist = dist.cuda()
                atoms_coord, atoms_token, atoms_index, batch_index = \
                    atoms_coord.cuda(), atoms_token.cuda(), atoms_index.cuda(), batch_index.cuda()
                generative_scores, context_scores = \
                    self.model(src, tgt, bond, dist, \
                               atoms_coord, atoms_token, atoms_index, batch_index)
                generative_scores_1, _ = \
                    self.model(src, tgt, bond, dist, \
                               atoms_coord, atoms_token, atoms_index, batch_index)
                # language modeling loss
                pred_token_logit = generative_scores.view(-1, generative_scores.size(2))
                pred_token_logit_1 = generative_scores_1.view(-1, generative_scores_1.size(2))
                gt_token_label = tgt[1:].view(-1)
                loss_token = self.criterion_tokens(pred_token_logit, pred_token_logit_1, gt_token_label)

                # loss for context alignment
                is_inferred = (gt_context_alignment.sum(dim=-1) == 0)
                gt_context_align_label = gt_context_alignment[~is_inferred].view(-1, gt_context_alignment.shape[-1])
                context_score = context_scores[-1]
                pred_context_align_logit = context_score[~is_inferred].view(-1, context_score.shape[-1])
                loss_context_align = self.criterion_context_align(pred_context_align_logit, gt_context_align_label)
                
                # add all loss
                loss = loss_token + loss_context_align

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
        writer = self.writer_dict['writer']
        global_steps = self.writer_dict['test_global_steps']
        if self.rank < 1:
            writer.add_scalar('test_loss', loss_per_word, global_steps)
            writer.add_scalar('test_accuracy', top1_accuracy, global_steps)
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
        return self.early_stop >= 7