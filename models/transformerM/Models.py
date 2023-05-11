''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer,DecoderLayer, GaussianLayer, NonLinear
import  torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=256):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=256, scale_emb=False,
            src_word_emb=None):

        super().__init__()

        if src_word_emb is not None:
            self.src_word_emb = src_word_emb
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.gbf = GaussianLayer(d_model)
        self.gbf_proj = NonLinear(d_model, n_head)
        self.rate1 = torch.nn.Parameter(torch.rand(1))
        self.rate2 = torch.nn.Parameter(torch.rand(1))

    def forward(self, src_seq, src_mask, src_pos, src_z, src_index, src_atoms, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.position_enc(enc_output)

        split_sizes = torch.bincount(src_index)
        max_node = max(split_sizes)
        src_pos_split = torch.split_with_sizes(src_pos, split_sizes.tolist())
        src_pos_batch = torch.zeros((src_seq.shape[0], max_node, 3), device=src_seq.device)
        for i in range(src_pos_batch.shape[0]):
            src_pos_batch[i,:split_sizes[i]] = src_pos_split[i]
        pos_mask = src_pos_batch.eq(0).all(dim=-1)
        delta_pos = src_pos_batch.unsqueeze(1) - src_pos_batch.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, max_node, max_node)
        edge_feature = self.gbf(dist, torch.zeros_like(dist).long()) # ([bs, n_node, n_node, k])
        edge_feature = edge_feature.masked_fill(
            pos_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )
        enc_output_pos = edge_feature.sum(dim=-2)
        pos_bias = torch.zeros_like(enc_output)
        for i in range(pos_bias.shape[0]):
            index = src_atoms[src_index==i]
            pos_bias[i, index] = enc_output_pos[i, :len(index)]

        enc_output = self.rate1 * enc_output + self.rate2 * pos_bias
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        attn_bias_3d = self.gbf_proj(edge_feature) # ([bs, n_node, n_node, rpe_heads])
        attn_bias_3d = attn_bias_3d.permute(0, 3, 1, 2).contiguous()

        bias_3d = torch.zeros((attn_bias_3d.shape[:2]+(src_seq.shape[1],)*2), device=src_seq.device)

        for i in range(bias_3d.shape[0]):
            index = src_atoms[src_index==i]
            for m,j in enumerate(index):
                bias_3d[i,:,j,index] = attn_bias_3d[i,:,m,:len(index)]

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask, attn_bias=bias_3d)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, src_mask


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=256, dropout=0.1, scale_emb=False,
            trg_word_emb=None):

        super().__init__()

        if trg_word_emb is not None:
            self.trg_word_emb = trg_word_emb
        else:
            self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class TransformerM(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj', share_vocab=False):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        if share_vocab:
            self.word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=src_pad_idx)
        else:
            self.word_emb = None

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb,
            src_word_emb=self.word_emb
            )

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb,
            trg_word_emb=self.word_emb
            )

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq, src_pos, src_z, src_index, src_atoms):
        """
        Args:
            src_seq: (N,S)
            trg_seq: (N,T)
        """
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        #print("encoder input shape:",src_seq.shape,src_mask.shape)#(N,S) (N,1,S)
        enc_output, src_mask, *_ = self.encoder(src_seq, src_mask, src_pos, src_z, src_index, src_atoms)
        #print("decoder input shape:", trg_seq.shape, trg_mask.shape,enc_output.shape,src_mask.shape)  # (N,T), (N,T,T), (N,S,E), (N,1,S)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        #print("decoder output shape", dec_output.shape)#(N,T,E)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        #print("transformer out shape:",seq_logit.shape)#(N,T,V)
        return seq_logit.view(-1, seq_logit.size(2))

    def forward_encoder(self,src_seq, src_pos, src_z, src_index, src_atoms):
        """
        一次encoder操作
        Args:
            src_seq: dim=(N,S), dtype=long_tensor
        Return:
            enc_output: dim=(N,S,E) , dtype= float_tensor
            src_mask: dim=(N,1,S), dtype =
        """
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        enc_output, src_mask, *_ = self.encoder(src_seq, src_mask, src_pos, src_z, src_index, src_atoms)
        return enc_output,src_mask

    def forward_decoder(self,trg_seq,enc_output,src_mask,temperature: float = 1.0):
        """重新封装的transformer deconder,每次只预测一个token"""
        trg_mask = get_subsequent_mask(trg_seq)
        #print("decoder input shape:", trg_seq.shape, trg_mask.shape, enc_output.shape,src_mask.shape)  # (N,T), (N,T,T), (N,S,E), (N,1,S)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        #print("decoder output shape", dec_output.shape)  # (N,T,E)
        dec_output = dec_output.div_(temperature)
        dec_output = self.trg_word_prj(dec_output)
        dec_output *= self.d_model ** -0.5
        dec_output = F.log_softmax(dec_output, dim=-1)
        #dec_output = dec_output - torch.tensor(4.127)
        return  dec_output # (N,T,V)