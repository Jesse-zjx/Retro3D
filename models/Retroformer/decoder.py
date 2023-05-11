import torch
import torch.nn as nn
from .module import MultiHeadAttention, PositionwiseFeedForward
from .module import LayerNorm
import numpy as np


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_inner, dropout, embeddings, self_attn_modules):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        context_attn_modules = nn.ModuleList(
            [MultiHeadAttention(n_head, d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_head, d_inner, dropout, self_attn_modules[i], context_attn_modules[i])
             for i in range(num_layers)]
        )
        self.layer_norm_0 = LayerNorm(d_model)
        self.layer_norm_1 = LayerNorm(d_model)

    def forward(self, src, tgt, memory_bank, nonreactive_mask=None, state_cache=None, step=None):
        """
        :param src:
        :param tgt:
        :param memory_bank:
        :param nonreactive_mask:
        :param state_cache:
        :param step:
        :return:
        """
        if nonreactive_mask is not None:
            nonreactive_mask[0] = False
        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        s_bsz, s_len = src_words.size()
        t_bsz, t_len = tgt_words.size()

        out_puts = []
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3
        if step is not None:
            tgt_words = tgt[-1].unsqueeze(0).transpose(0, 1)
            t_bsz, t_len = tgt_words.size()
        output = emb.transpose(0, 1).contiguous()
        # encoder的输出
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        padding_idx = self.embeddings.padding_idx

        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1).expand(s_bsz, t_len, s_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1).expand(t_bsz, t_len, t_len)
        nonreactive_mask_input = nonreactive_mask.transpose(0, 1) if nonreactive_mask is not None else None
        top_context_attns = []
        for i in range(self.num_layers):
            layer_input = None
            layer_cache = {'self_keys': None,
                           'self_values': None,
                           'memory_keys': None,
                           'memory_values': None}
            if state_cache is not None:
                layer_cache = state_cache.get('layer_cache_{}'.format(i), layer_cache)
                layer_input = state_cache.get('layer_input_{}'.format(i), layer_input)
            output, top_context_attn, all_input = self.decoder_layers[i](output, src_memory_bank, src_pad_mask,
                                                                         tgt_pad_mask, layer_input=layer_input,
                                                                         layer_cache=layer_cache,
                                                                         nonreactive_mask_input=nonreactive_mask_input)
            top_context_attns.append(top_context_attn)
            if state_cache is not None:
                state_cache['layer_cache_{}'.format(i)] = layer_cache
                state_cache['layer_input_{}'.format(i)] = all_input
        output = self.layer_norm_1(output)
        outputs = output.transpose(0, 1).contiguous()
        return outputs, top_context_attns


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout, self_attn, context_attn):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.context_attn = context_attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(5000)
        self.register_buffer('mask', mask)

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask

    def forward(self, tgt_emb, enc_out, src_pad_mask, tgt_pad_mask, nonreactive_mask_input=None, layer_input=None,
                layer_cache=None):
        # inputs (`FloatTensor`): `[batch_size x tgt_len x model_dim]`
        # memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
        # src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
        # infer_decision_input (`LongTensor`): `[batch_size x tgt_len]`
        # nonreactive_mask_input (`BoolTensor`): `[batch_size x src_len]`
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)
        tgt_norm = self.layer_norm_1(tgt_emb)
        all_input = tgt_norm
        if layer_input is not None:
            all_input = torch.cat((layer_input, tgt_norm), dim=1)
            dec_mask = None
        # self_attention
        query, self_attn, _ = self.self_attn(all_input, all_input, tgt_norm, mask=dec_mask, type='self',
                                             layer_cache=layer_cache)
        query = self.drop(query) + tgt_emb
        query_norm = self.layer_norm_2(query)

        mid, context_attn, _ = self.context_attn(enc_out, enc_out, query_norm, mask=src_pad_mask,
                                                 additional_mask=nonreactive_mask_input, type='context',
                                                 layer_cache=layer_cache)
        """ 感觉应该是+query_norm """
        output = self.feed_forward(self.drop(mid) + query)
        return output, context_attn, all_input
