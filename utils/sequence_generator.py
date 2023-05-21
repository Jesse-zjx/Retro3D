# Original code from fairseq
# Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to https://arxiv.org/abs/1609.08144.
import math
import sys
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from .search import BeamSearch


class SequenceGenerator(nn.Module):
    def __init__(
            self,
            config,
            models,
            data,
            beam_size=1,
            max_len_a=0,
            max_len_b=162,
            max_len=0,
            min_len=4,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=1.0,
            temperature=1.0,
            match_source_len=False,
            eos=None,
            search_strategy=BeamSearch
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            max_len (int, optional): the maximum length of the generated output
                (not including end-of-sentence)
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        self.config = config
        self.model = models
        self.tgt_t2i = data.tgt_t2i
        self.pad = self.tgt_t2i['<pad>']
        self.unk = [self.tgt_t2i['<UNK>'], self.tgt_t2i['<unk>']] + \
                [self.tgt_t2i['<RX_{}>'.format(i)] for i in range(1, 11)]
        self.bos = self.tgt_t2i['<sos>']
        self.eos = self.tgt_t2i['<eos>']

        self.vocab_size = len(self.tgt_t2i)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.max_len = max_len

        self.normalize_scores = normalize_scores
        # self.len_penalty = len_penalty
        # self.len_penalty = 0.1
        self.alp = 0.5
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len

        assert temperature > 0, "--temperature must be greater than 0"

        # TODO
        self.search = BeamSearch(self.pad, self.eos, len(self.tgt_t2i))
        self.model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(self, sample):
        """Generate a batch of translations.

        Args:
            sample : batch
        """
        return self._generate(sample)

    def _generate(self, sample):

        src_tokens, bond, dist = sample
        # length of the source text being the character length except EndOfSentence and pad
        # 计算除了eos和pad的原始长度
        src_lengths = (
            (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        )
        # print(src_tokens)
        # print(src_lengths)
        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        max_len = int(self.max_len_a * src_len + self.max_len_b)
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # print(max_len, self.min_len)
        # compute the encoder output for each beam
        encoder_outs, nonreacrive_mask = self.model.forward_encoder(src_tokens.transpose(0, 1), bond, dist)
        # encoder_outs, src_masks = self.model.forward_encoder(src_tokens.transpose(0, 1))

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        # print('new order',new_order)
        encoder_outs = self.reorder_encoder_out(encoder_outs.transpose(0, 1), new_order)
        nonreacrive_mask = self.reorder_encoder_out(nonreacrive_mask.transpose(0, 1), new_order)
        src_tokens_expand = self.reorder_encoder_out(src_tokens, new_order)
        encoder_outs = encoder_outs.transpose(0, 1)
        nonreacrive_mask = nonreacrive_mask.transpose(0, 1)
        # print(encoder_outs.shape)#N*beam_size,S,E
        # print(encoder_outs[0]==encoder_outs[1])#True
        # print(src_masks.shape)#(N*beam_size,1,S)
        # print(src_masks[0]==src_masks[1])#True
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.bos

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences List[List[Dict[str, Tensor]]]
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        # print(bbsz_offsets)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None
        # TODO add dataset level batch idxs
        original_batch_idxs = torch.arange(0, bsz).type_as(tokens)
        # print(original_batch_idxs)
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            # print('step',step)
            # print(batch_idxs)#
            # print(reorder_state)#
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                encoder_outs = self.reorder_encoder_out(
                    encoder_outs.transpose(0, 1), reorder_state
                )
                encoder_outs = encoder_outs.transpose(0, 1)
                nonreacrive_mask = self.reorder_encoder_out(nonreacrive_mask.transpose(0, 1), reorder_state)
                nonreacrive_mask = nonreacrive_mask.transpose(0, 1)
                src_tokens_expand = self.reorder_encoder_out(src_tokens_expand, reorder_state)
                # print(encoder_outs.shape)#N*beam_size,S,E
                # print(encoder_outs[0]==encoder_outs[1])#True
                # print(src_masks.shape)#(N*beam_size,1,S)
                # print(src_masks[0]==src_masks[1])#True
            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                # print('size : [{}]'.format(src_tokens_expand.size()))
                lprobs = self.model.forward_decoder(
                    src_tokens_expand.transpose(0, 1),
                    tokens[:, : step + 1].transpose(0, 1),
                    encoder_outs,
                    nonreacrive_mask
                ).transpose(0, 1).contiguous()[:, -1, :]
            # print(tokens[:, : step + 1].shape)#(N*bsz,T)
            # print("lprobs shape", lprobs.shape)#(N*bsz,V)

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            if step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)
            # print(eos_bbsz_idx)
            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            # print(bbsz_offsets)
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)

            # batch_eos_mask = (eos_mask.sum(dim=1) >= beam_size)
            # eos_mask = eos_mask * batch_eos_mask.unsqueeze(dim=1)
            # print(eos_mask.sum(dim=1))

            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            #
            # print(eos_mask.sum(dim=1))
            # print(eos_mask.shape)
            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )
            # print(eos_bbsz_idx)
            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )
                max_ctn_score = self.f_max_ctn(cand_scores, eos_mask)
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    src_lengths,
                    max_len,
                    max_ctn_score
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def f_min(self, curr_final_beam, beam_size):
        min_score = math.inf
        min_ind = -1
        for i in range(beam_size):
            if curr_final_beam[i]["score"] < min_score:
                min_score = curr_final_beam[i]["score"]
                min_ind = i
        return min_score, min_ind

    def f_max_ctn(self, cand_scores, eos_mask):
        bsz = cand_scores.size()[0]
        beam_2 = cand_scores.size()[1]
        re = torch.zeros(bsz)

        for i in range(bsz):
            for j in range(beam_2):
                if not eos_mask[i][j]:
                    re[i] = cand_scores[i][j]
                    break
        return re

    def finalize_hypos(
            self,
            step: int,
            bbsz_idx,
            eos_scores,
            tokens,
            scores,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            beam_size: int,
            src_lengths,
            max_len: int,
            max_ctn_score,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        # print(bbsz_idx)
        tokens_clone = tokens.index_select(0, bbsz_idx)[
                       :, 1: step + 2
                       ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            # eos_scores /= (step + 1) ** self.len_penalty
            eos_scores /= (1 + step / 6) ** self.alp

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)
        cum_fin_tensor = torch.tensor(cum_unfin, dtype=torch.int).to(bbsz_idx)

        unfin_idx = torch.div(bbsz_idx, beam_size, rounding_mode="trunc")
        sent = unfin_idx + torch.index_select(cum_fin_tensor, 0, unfin_idx)

        # Create a set of "{sent}{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # For every finished beam item
        # sentence index in the current (possibly reduced) batch
        seen = (sent << 32) + unfin_idx
        unique_seen: List[int] = torch.unique(seen).tolist()

        if self.match_source_len:
            condition = step > torch.index_select(src_lengths, 0, unfin_idx)
            eos_scores = torch.where(condition, torch.tensor(-math.inf), eos_scores)
        sent_list: List[int] = sent.tolist()
        for i in range(bbsz_idx.size()[0]):
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent_list[i]]) < beam_size:
                finalized[sent_list[i]].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": eos_scores[i],
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )
            else:
                min_score, min_ind = self.f_min(finalized[sent_list[i]], beam_size)
                if min_score < eos_scores[i]:
                    finalized[sent_list[i]][min_ind]["tokens"] = tokens_clone[i]
                    finalized[sent_list[i]][min_ind]["score"] = eos_scores[i]
                    finalized[sent_list[i]][min_ind]["alignment"] = torch.empty(0)
                    finalized[sent_list[i]][min_ind]["positional_scores"] = pos_scores[i]

        newly_finished: List[int] = []
        for unique_s in unique_seen:
            # check termination conditions for this sentence
            unique_sent: int = unique_s >> 32
            unique_unfin_idx: int = unique_s - (unique_sent << 32)

            if not finished[unique_sent] and self.is_finished(
                step, unique_unfin_idx, max_len, finalized[unique_sent], beam_size, max_ctn_score
            ):
                finished[unique_sent] = True
                newly_finished.append(unique_unfin_idx)

        return newly_finished

    def is_finished(
            self,
            step: int,
            unique_sent: int,
            max_len: int,
            finalized_sent,
            beam_size: int,
            max_ctn_score
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        # assert finalized_sent_len <= beam_size
        # if finalized_sent_len == beam_size or step == max_len:
        #     return True
        # return False
        if len(finalized_sent) == beam_size:
            min_eos_score, _ = self.f_min(finalized_sent, beam_size)
            max = (max_ctn_score[unique_sent] / (((5 + max_len) / 6) ** self.alp)).to(min_eos_score)
            if max < min_eos_score:
                return True
        if step == max_len:
            return True
        return False

    def reorder_encoder_out(self, encoder_outputs, new_order):
        return encoder_outputs.index_select(0, new_order)
