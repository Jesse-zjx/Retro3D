import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__(beta=beta, threshold=threshold)

    def forward(self, x):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(x, self.beta, self.threshold) - sp0


class MultiHeadAttention(nn.Module):
    def __init__(self, head_cnt, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head_cnt == 0
        self.dim_per_head = d_model // head_cnt
        self.head_cnt = head_cnt
        self.d_model = d_model
        self.linear_keys = nn.Linear(d_model, head_cnt * self.dim_per_head)
        self.linear_query = nn.Linear(d_model, head_cnt * self.dim_per_head)
        self.linear_value = nn.Linear(d_model, head_cnt * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_attn = nn.Dropout(dropout)
        self.final_linear = nn.Linear(d_model, d_model)
        self.edge_project = nn.Sequential(nn.Linear(d_model, d_model), SSP(), nn.Linear(d_model, d_model // 2))
        self.edge_update = nn.Sequential(nn.Linear(d_model * 2, d_model), SSP(), nn.Linear(d_model, d_model))

    def forward(self, key, value, query, mask, additional_mask=None, edge_feature=None,
                pair_indices=None):
        global q_project, key_shaped, value_shaped
        bsz = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_cnt
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(bsz, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(bsz, -1, head_count * dim_per_head)

        # linear for k,q,v
        k_project = self.linear_keys(key)
        q_project = self.linear_query(query)
        v_project = self.linear_value(value)
        key_shaped = shape(k_project)
        value_shaped = shape(v_project)
        query_shaped = shape(q_project)
        # (bsz,head,len,dim_per_head)
        key_len = key_shaped.size(2)
        query_len = query_shaped.size(2)

        if edge_feature is not None:
            # encoder
            edge_feature_shaped = self.edge_project(edge_feature).view(-1, head_count // 2, dim_per_head)
            key_shaped_local = key_shaped[pair_indices[0], head_count // 2:, pair_indices[2]]
            query_shaped_local = query_shaped[pair_indices[0], head_count // 2:, pair_indices[1]]
            value_shaped_local = value_shaped[:, head_count // 2:]

            key_shaped_local = key_shaped_local * edge_feature_shaped
            query_shaped_local = query_shaped_local / math.sqrt(dim_per_head)
            scores_local = torch.matmul(query_shaped_local.unsqueeze(2), key_shaped_local.unsqueeze(3)).view(
                edge_feature.shape[0], head_count // 2)
            score_expand_local = scores_local.new_full(
                (value.shape[0], value.shape[1], value.shape[1], head_count // 2), -float('inf')
            )
            score_expand_local[pair_indices] = scores_local
            score_expand_local = score_expand_local.transpose(1, 3).transpose(2, 3)
            attn_local = self.softmax(score_expand_local)
            attn_local = attn_local.masked_fill(score_expand_local < -10000, 0)
            drop_attn_local = self.drop_attn(attn_local)
            local_context = torch.matmul(drop_attn_local, value_shaped_local)

            query_shaped_global = query_shaped[:, :head_count // 2]
            key_shaped_global = key_shaped[:, :head_count // 2]
            value_shaped_global = value_shaped[:, :head_count // 2]

            query_shaped_global = query_shaped_global / math.sqrt(dim_per_head)
            score_global = torch.matmul(query_shaped_global, key_shaped_global.transpose(2, 3))
            top_score = score_global.view(bsz, score_global.shape[1], query_len, key_len)[:, 0, :, :].contiguous()

            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(score_global).clone()
                score_global = score_global.masked_fill(mask, -1e18)
            attn = self.softmax(score_global)
            drop_attn = self.drop_attn(attn)
            global_context = torch.matmul(drop_attn, value_shaped_global)

            context = torch.cat([global_context, local_context], dim=-1)
            context = unshape(context)
        else:
            # normal encoder_decoder
            query_shaped = query_shaped / math.sqrt(dim_per_head)
            scores = torch.matmul(query_shaped, key_shaped.transpose(2, 3))
            top_score = scores.view(bsz, scores.shape[1],
                                    query_len, key_len)[:, 0, :, :].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(scores).clone()
                scores = scores.masked_fill(mask, -1e18)
            attn = self.softmax(scores)
            drop_attn = self.drop_attn(attn)
            context = torch.matmul(drop_attn, value_shaped)
            context = unshape(context)
        output = self.final_linear(context)

        if edge_feature is not None:
            node_feature_updated = output
            node_features = torch.cat([node_feature_updated[pair_indices[0], pair_indices[1]],
                                       node_feature_updated[pair_indices[0], pair_indices[2]]], dim=-1)
            edge_feature_updated = self.edge_update(node_features)
            return output, top_score, edge_feature_updated
        else:
            return output, top_score, None


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=7):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Linear(edge_types, 1)
        self.bias = nn.Linear(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def gaussian(self, x, mean, std):
        pi = 3.14159
        a = (2*pi) ** 0.5
        return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = x.unsqueeze(-1)
        x = (mul * x + bias)
        x = x.expand(-1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return self.gaussian(x.float(), mean, std).type_as(self.means.weight)

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x