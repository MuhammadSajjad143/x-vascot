"""LSTM baseline and the VASCOT transformer.

Architecture is unchanged from the notebook. Two behaviours are parameterised
because the notebook and the paper disagree about them:

``window_agg``
    ``mean_pooled`` averages the pooled hidden representation across sliding
    windows (what the notebook codes). ``mean_prob`` averages the per-window
    class probabilities (what the paper describes in Sec. III-B / IV-D).

Sliding windows only ever engage when a sequence is longer than ``max_length``.
In the notebook they never did, because the data pipeline had already cut every
contract to exactly 2048 tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to embeddings."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class AttentionPooling(nn.Module):
    """Learned importance over the sequence; reduces (B, L, D) -> (B, D)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = torch.tanh(self.attn(x))
        weights = F.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)


class TransformerEncoderBlock(nn.Module):
    """LayerNorm -> MHA -> residual -> LayerNorm -> Conv1d FFN -> residual."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(embed_dim, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, embed_dim, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=return_attn, average_attn_weights=False)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
        return (x, attn_weights) if return_attn else x


class LSTMModel(nn.Module):
    """Tann et al. (2018) sequence classifier: Embedding -> Dropout -> LSTM -> FC."""

    def __init__(self, num_words, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(num_words, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class VASCOTModel(nn.Module):
    """Embedding -> PosEnc -> TransformerEncoder -> AttnPool -> Dense -> Linear(2)."""

    def __init__(self, config: dict, window_agg: str = "mean_pooled"):
        super().__init__()
        self.config = config
        self.window_agg = window_agg
        d = config["embedding_dim"]
        self.embedding = nn.Embedding(config["num_words"], d, padding_idx=0)
        self.pos_enc = PositionalEncoding(d, config["max_length"], config["dropout"])
        self.encoder = TransformerEncoderBlock(d, config["num_heads"],
                                               config["ff_dim"], config["dropout"])
        self.pool = AttentionPooling(d)
        self.drop1 = nn.Dropout(config["dropout"])
        self.dense = nn.Linear(d, d)
        self.drop2 = nn.Dropout(config["dropout"])
        self.out = nn.Linear(d, config.get("output_dim", 2))

    # ── head applied to a pooled representation ──────────────────────────────
    def classify(self, pooled):
        return self.out(self.drop2(F.relu(self.dense(self.drop1(pooled)))))

    def encode_embedded(self, emb, return_attn: bool = False):
        """Run the post-embedding stack on an embedding tensor (1 window)."""
        x = self.pos_enc(emb)
        if return_attn:
            x, attn = self.encoder(x, return_attn=True)
            return self.pool(x), attn
        return self.pool(self.encoder(x))

    def aggregate(self, pooled_list):
        """Model output from per-window pooled vectors (the value ``forward`` returns).

        One window: the classifier logit. Several: ``mean_prob`` averages the
        per-window probabilities and returns their log (so the loss and argmax
        work unchanged); ``mean_pooled`` classifies the mean pooled vector.
        """
        if len(pooled_list) == 1:
            return self.classify(pooled_list[0])
        if self.window_agg == "mean_prob":
            probs = torch.stack([F.softmax(self.classify(p), dim=1)
                                 for p in pooled_list], 0).mean(0)
            return torch.log(probs.clamp_min(1e-12))
        return self.classify(torch.stack(pooled_list, 0).mean(0))

    def window_starts(self, L: int) -> list[int]:
        """Start offsets of the sliding windows covering a length-L sequence."""
        W = self.config["max_length"]
        if L <= W:
            return [0]
        step = int(W * self.config["sliding_window_ratio"])
        starts = list(range(0, L - W + 1, step))
        if starts[-1] + W < L:
            starts.append(L - W)          # ensure the tail is covered
        return starts

    def forward(self, x, return_attn: bool = False):
        B, L = x.shape
        W = self.config["max_length"]

        if L <= W:
            emb = self.embedding(x)
            if return_attn:
                pooled, attn = self.encode_embedded(emb, return_attn=True)
                return self.classify(pooled), attn
            return self.classify(self.encode_embedded(emb))

        starts = self.window_starts(L)
        pooled_list, attn_list = [], []
        for s in starts:
            emb = self.embedding(x[:, s:s + W])
            if return_attn:
                p, a = self.encode_embedded(emb, return_attn=True)
                attn_list.append(a)
            else:
                p = self.encode_embedded(emb)
            pooled_list.append(p)

        logit = self.aggregate(pooled_list)

        if return_attn:
            return logit, torch.stack(attn_list, 0).mean(0)
        return logit
