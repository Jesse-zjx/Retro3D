import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import MultiHeadAttention, PositionwiseFeedForward, LayerNorm
from .module import GaussianLayer, NonLinear
from .ComENet import ComENet


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
    def __init__(self, num_layers, d_model, n_head, d_inner, dropout, embeddings, attn_modules):
        super(TransformerEncoder, self).__init__()
        self.embeddings = embeddings
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_head, d_inner, dropout, attn_modules[i])
             for i in range(num_layers)]
        )
        self.layer_morm = LayerNorm(d_model)
        self.gbf = GaussianLayer(d_model)
        self.gbf_proj = NonLinear(d_model, d_model)
        self.comenet = ComENet(hidden_channels=d_model, middle_channels=d_model, out_channels=d_model, emb=embeddings.token)
        self.rate1 = torch.nn.Parameter(torch.rand(1))
        self.rate2 = torch.nn.Parameter(torch.rand(1))

    def forward(self, x, bond=None, dist=None, atoms_coord=None, atoms_token=None, atoms_index=None, batch_index=None):
        emb = self.embeddings(x)
        out = emb.transpose(0, 1).contiguous()

        out_pos = self.comenet(atoms_coord, atoms_token, batch_index)
        pos_bias = torch.zeros_like(out)
        for i in range(pos_bias.shape[0]):
            feature = out_pos[batch_index==i]
            index = atoms_index[batch_index==i]+1   #+1 for '<RX_*>/<UNK>'
            pos_bias[i, index] = feature
        out = self.rate1 * out + self.rate2 * pos_bias

        # if bond is not None:
        #     # 找到有feature的bond
        #     pair_indices = torch.where(bond.sum(-1) > 0)
        #     valid_bond = bond[bond.sum(-1) > 0]
        #     edge_feature = self.embeddings_bond(valid_bond.float())
        # else:
        #     pair_indices, edge_feature = None, None

        if dist is not None and bond is not None:
            pair_indices = torch.where(dist != 0)
            valid_dist = dist[dist != 0]
            valid_bond = bond[dist != 0]
            edge_feature = self.gbf(valid_dist, valid_bond.float())
            edge_feature = self.gbf_proj(edge_feature)
        else:
            pair_indices, edge_feature = None, None

        src = x.transpose(0, 1)
        bsz, b_len = src.size()
        padding_idx = self.embeddings.padding_idx
        mask = src.data.eq(padding_idx).unsqueeze(1).expand(bsz, b_len, b_len)
        for encoder_layer in self.encoder_layers:
            out, attn, edge_feature = encoder_layer(out, mask, edge_feature, pair_indices)
        out = self.layer_morm(out)
        out = out.transpose(0, 1).contiguous()
        edge_out = self.layer_morm(edge_feature) if edge_feature is not None else None
        return out, edge_out
