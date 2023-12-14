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
        self.d_model = d_model

        self.src_embedding = Embedding(vocab_size=n_src_vocab + 1, embed_size=d_model, padding_idx=src_pad_idx)
        if shared_embed:
            assert n_src_vocab == n_trg_vocab and src_pad_idx == tgt_pad_idx
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = Embedding(vocab_size=n_trg_vocab + 1, embed_size=d_model, padding_idx=tgt_pad_idx)

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
                                          attn_modules=multihead_attn_modules_en)
        self.decoder = TransformerDecoder(num_layers=n_dec_layers, d_model=d_model, n_head=n_head, d_inner=d_inner,
                                          dropout=dropout, embeddings=self.tgt_embedding,
                                          self_attn_modules=multihead_attn_modules_de)
        self.projection = nn.Sequential(nn.Linear(d_model, n_trg_vocab), nn.LogSoftmax(dim=-1))

    def forward(self, src, tgt, bond=None, dist=None, atoms_coord=None, atoms_token=None, atoms_index=None, batch_index=None):
        encoder_out, edge_feature = self.encoder(src, bond, dist, atoms_coord, atoms_token, atoms_index, batch_index)
        decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out)
        logit = self.projection(decoder_out)
        return logit, top_aligns

    def forward_encoder(self, src, bond, dist, atoms_coord, atoms_token, atoms_index, batch_index):
        encoder_out, edge_feature = self.encoder(src, bond, dist, atoms_coord, atoms_token, atoms_index, batch_index)
        return encoder_out

    def forward_decoder(self, src, tgt, encoder_out):
        """重新封装的transformer deconder,每次只预测一个token"""
        dec_output, _ = self.decoder(src, tgt, encoder_out)
        return self.projection(dec_output)
