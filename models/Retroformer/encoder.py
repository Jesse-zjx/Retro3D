import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import MultiHeadAttention, PositionwiseFeedForward, LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_inner, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_atten = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, edge_feature, pair_indices):
        input_norm = self.layer_norm(x)
        context, attn, edge_feature_updated = self.self_atten(input_norm, input_norm, input_norm, mask=mask,
                                                              edge_feature=edge_feature, pair_indices=pair_indices)
        out = self.dropout(context) + x
        if edge_feature is not None:
            edge_feature = self.layer_norm(edge_feature + edge_feature_updated)
        return self.feed_forward(out), attn, edge_feature


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, n_head, d_inner, dropout, embeddings, embeddings_bond, attn_modules):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.embeddings_bond = embeddings_bond
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_head, d_inner, dropout, attn_modules[i])
             for i in range(num_layers)]
        )
        self.layer_morm = LayerNorm(d_model)

    def forward(self, x, bond=None):
        global node_feature
        emb = self.embeddings(x)
        # 转成batch size first
        out = emb.transpose(0, 1).contiguous()
        if bond is not None:
            # 找到有feature的bond
            pair_indices = torch.where(bond.sum(-1) > 0)
            valid_bond = bond[bond.sum(-1) > 0]
            edge_feature = self.embeddings_bond(valid_bond.float())
        else:
            pair_indices, edge_feature = None, None

        src = x.transpose(0, 1)
        bsz, b_len = src.size()
        padding_idx = self.embeddings.padding_idx
        mask = src.data.eq(padding_idx).unsqueeze(1).expand(bsz, b_len, b_len)
        for i in range(self.num_layers):
            out, attn, edge_feature = self.encoder_layers[i](out, mask, edge_feature, pair_indices)
        out = self.layer_morm(out)
        out = out.transpose(0, 1).contiguous()
        edge_out = self.layer_morm(edge_feature) if edge_feature is not None else None
        return out, edge_out
