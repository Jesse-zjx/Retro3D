import torch.nn as nn
import torch
from .embedding import Embedding
from .module import MultiHeadAttention
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Retroformer(nn.Module):
    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, tgt_pad_idx, d_model, d_inner, 
                n_enc_layers, n_dec_layers, n_head, dropout, shared_embed, shared_encoder=False):
        super(Retroformer, self).__init__()
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.shared_encoder = shared_encoder
        if shared_embed:
            assert n_src_vocab == n_trg_vocab and src_pad_idx == tgt_pad_idx
            self.src_embedding = self.tgt_embedding = Embedding(vocab_size=n_src_vocab + 1, embed_size=d_model,
                                                                padding_idx=src_pad_idx)
        else:
            self.src_embedding = Embedding(vocab_size=n_src_vocab + 1, embed_size=d_model, padding_idx=src_pad_idx)
            self.tgt_embedding = Embedding(vocab_size=n_trg_vocab + 1, embed_size=d_model, padding_idx=tgt_pad_idx)
        self.bond_embedding = nn.Linear(7, d_model)

        multihead_attn_modules_en = nn.ModuleList(
            [MultiHeadAttention(n_head, d_model, dropout=dropout)
             for _ in range(n_enc_layers)]
        )
        if shared_encoder:
            assert n_enc_layers == n_dec_layers
            multihead_attn_modules_de = multihead_attn_modules_en
        else:
            multihead_attn_modules_de = nn.ModuleList(
                [MultiHeadAttention(n_head, d_model, dropout=dropout)
                 for _ in range(n_dec_layers)]
            )
        self.encoder = TransformerEncoder(num_layers=n_enc_layers, d_model=d_model, n_head=n_head, d_inner=d_inner,
                                          dropout=dropout, embeddings=self.src_embedding,
                                          embeddings_bond=self.bond_embedding, attn_modules=multihead_attn_modules_en)
        self.decoder = TransformerDecoder(num_layers=n_dec_layers, d_model=d_model, n_head=n_head, d_inner=d_inner,
                                          dropout=dropout, embeddings=self.tgt_embedding,
                                          self_attn_modules=multihead_attn_modules_de)
        self.atom_rc_identifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.bond_rc_identifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.projection = nn.Sequential(nn.Linear(d_model, n_trg_vocab), nn.LogSoftmax(dim=-1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, bond=None, dist=None, teacher_mask=None):
        encoder_out, edge_feature = self.encoder(src, bond, dist)
        # 预测在编码器特征序列中的反应位置
        # atom_rc_scores = self.atom_rc_identifier(encoder_out)
        atom_rc_scores = None
        # bond_rc_scores = self.bond_rc_identifier(edge_feature) if edge_feature is not None else None
        bond_rc_scores = None
        teacher_mask = None
        if teacher_mask is None:
            # student_mask = self.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, None)# student_mask.clone()
        else:
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, None)# teacher_mask.clone()
        logit = self.projection(decoder_out)
        return logit, atom_rc_scores, bond_rc_scores, top_aligns

    def forward_encoder(self, src, bond, dist):
        encoder_out, edge_feature = self.encoder(src, bond, dist)
        # atom_rc_scores = self.atom_rc_identifier(encoder_out)
        # bond_rc_scores = self.bond_rc_identifier(edge_feature) if edge_feature is not None else None
        # nonreactive_mask = self.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)
        nonreactive_mask = None
        return encoder_out, nonreactive_mask

    def forward_decoder(self, src, tgt, encoder_out, student_mask):
        """重新封装的transformer deconder,每次只预测一个token"""
        dec_output, _ = self.decoder(src, tgt, encoder_out, student_mask)
        return self.projection(dec_output)

    @staticmethod
    def infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores):
        atom_rc_scores = atom_rc_scores.squeeze(2)
        if bond_rc_scores is not None:
            bond_rc_scores = bond_rc_scores.squeeze(1)
            bond_indicator = torch.zeros((bond.shape[0], bond.shape[1], bond.shape[2])).bool().to(bond.device)
            bond_indicator[bond.sum(-1) > 0] = (bond_rc_scores > 0.5)
            # print(bond_indicator.size())
            result = (~(bond_indicator.sum(dim=1).bool()) + ~(bond_indicator.sum(dim=2).bool()) + (
                    atom_rc_scores.transpose(0, 1) < 0.5)).transpose(0, 1)
        else:
            result = (atom_rc_scores.transpose(0, 1) < 0.5).transpose(0, 1)
        return result
