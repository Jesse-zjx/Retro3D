import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, padding_idx=1):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=padding_idx)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        position_e = torch.zeros(max_len, d_model).float()
        position_e.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        position_e[:, 0::2] = torch.sin(position * div_term)
        position_e[:, 1::2] = torch.cos(position * div_term)
        position_e = position_e.unsqueeze(0)
        self.register_buffer('pe', position_e)

    def forward(self, x):
        return self.pe[:, :x.size(0)].transpose(0, 1)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1, padding_idx=1):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, padding_idx=padding_idx)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=512)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.padding_idx = padding_idx

    def forward(self, x, step=None):
        output = self.token(x) + self.position(x)
        if step is None:
            return self.dropout(output)
        else:
            return self.dropout(output)[step].unsqueeze(0)
