# Original code from https://github.com/pytorch/fairseq/blob/main/fairseq/search.py
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class Search(nn.Module):
    def __init__(self, trg_pad_idx,trg_eos_idx,num_vocab):
        super().__init__()
        self.pad = trg_pad_idx
        self.eos = trg_eos_idx
        self.vocab_size = num_vocab
        self.src_lengths = torch.tensor(-1)

    def step(
        self, step, lprobs, scores, prev_output_tokens=None, original_batch_idxs=None
    ):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point
            prev_output_tokens: (bsz x step)
                the previously generated oputput tokens
            original_batch_idxs: (bsz)
                the tensor with the batch indices, in the range [0, bsz)
                this is useful in case there has been applied a re-ordering
                and we need to know the orignal indices

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

class BeamSearch(Search):
    def __init__(self, trg_pad_idx,trg_eos_idx,num_vocab):
        super().__init__(trg_pad_idx,trg_eos_idx,num_vocab)

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()
        # print("beam search step:", step)  # step
        # print("beam search lprobs:", lprobs.size())  # (N,beam_size,num_tokens)
        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:

            # print("beam search scores:", scores.size())  # (N,beam_size,step)
            # print("beam search prev output tokens:", prev_output_tokens.size())  # (N*beam_size,step+1)
            # print("beam search original idxs:", original_batch_idxs.size())  # (N)
            # make probs contain cumulative scores for each hypothesis
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(
                # Take the best 2 x beam_size predictions. We'll choose the first
                # beam_size of these which don't predict eos to continue with.
                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        # Project back into relative indices and beams
        # print(indices_buf.shape, vocab_size)
        beams_buf = torch.div(indices_buf, vocab_size, rounding_mode="trunc")
        indices_buf = indices_buf.fmod(vocab_size)

        # At this point, beams_buf and indices_buf are single-dim and contain relative indices
        # print(scores_buf.shape,indices_buf.shape,beams_buf.shape,'\n')
        # print(scores_buf, indices_buf, beams_buf)
        return scores_buf, indices_buf, beams_buf
