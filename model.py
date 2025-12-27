import torch
import torch.nn as nn
import numpy as np

from typing import TypedDict
from dataclasses import dataclass


class ScaledEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.scale: float = d_model**0.5
        self.embedding: nn.Embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        log_10000 = np.log(10000.0)

        div_term = torch.exp(
            -torch.arange(0, d_model, 2).float() * (log_10000 / d_model)
        )  # (d_model // 2, )
        encoding = position * div_term  # (max_len, d_model // 2)

        # dim1 here is batch_size. setting it to 1 so that broadcasting works with batches
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(encoding)
        pe[:, 0, 1::2] = torch.cos(encoding)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.shape[0]]
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        assert d_model % h == 0, "d_model is not perfectly divisible by h"

        self.h: int = h
        self.d_k: int = d_model // h
        self.sqrt_d_k: float = self.d_k**0.5

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,  # (batch_size, seq_len, d_model)
        k: torch.Tensor,  # same as above
        v: torch.Tensor,  # same as above
        mask: torch.Tensor | None,  # (batch_size, seq_len, seq_len)
    ) -> torch.Tensor:
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)  # same as above
        value = self.w_v(v)  # same as above

        batch_size, seq_len, _ = q.shape
        # split into h pieces
        # we have to do (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_k)
        # first         (batch_size, seq_len, d_model) --> (batch_size, seq_len, h, d_k)
        # then          (batch_size, seq_len, h, d_k).transpose(1, 2) --> (batch_size, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(
            1, 2
        )  # (batch_size, h, seq_len, d_k)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(
            1, 2
        )  # (batch_size, h, seq_len, d_k)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(
            1, 2
        )  # (batch_size, h, seq_len, d_k)

        attention_scores = (
            query @ key.transpose(-1, -2)
        ) / self.sqrt_d_k  # (batch_size, h, seq_len, seq_len)

        if mask is not None:
            attention_scores.masked_fill(mask == 0, 1e-9)

        attention_scores = torch.softmax(
            attention_scores, dim=-1
        )  # (batch_size, h, seq_len, seq_len)

        attention = attention_scores @ value  # (batch_size, h, seq_len, d_k)
        attention = (
            attention.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )  # (batch_size, seq_len, d_model)

        return self.w_o(attention)  # (batch_size, seq_len, d_model)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # self-attention
        attention_output = self.mha(x, x, x, mask)  # (batch_size, seq_len, d_model)
        attention_output = self.dropout1(attention_output)
        x = self.norm1(attention_output + x)

        ff_output = self.ff(x)  # (batch_size, seq_len, d_model)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(ff_output + x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_len: int,
        h: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # stack N encoder blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, h)
        self.mha = MultiHeadAttention(d_model, h)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, d_model)
        encoder_output: torch.Tensor,  # (batch_size, seq_len, d_model)
        src_mask: torch.Tensor | None,  # (batch_size, seq_len, seq_len)
        tgt_mask: torch.Tensor | None,  # (batch_size, seq_len, seq_len)
    ) -> torch.Tensor:

        # self attention
        masked_attention_output = self.masked_mha(
            x, x, x, tgt_mask
        )  # (batch_size, seq_len, d_model)
        masked_attention_output = self.dropout1(masked_attention_output)
        x = self.norm1(masked_attention_output + x)

        # cross attention
        attention_output = self.mha(
            x, encoder_output, encoder_output, src_mask
        )  # (batch_size, seq_len, d_model)
        attention_output = self.dropout2(attention_output)
        x = self.norm2(attention_output + x)

        ff_output = self.ff(x)  # (batch_size, seq_len, d_model)
        ff_output = self.dropout3(ff_output)
        x = self.norm3(ff_output + x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_len: int,
        h: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # stack N decoder blocks
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, h, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x


@dataclass
class TransformerConfig:
    d_model: int = 512
    vocab_size_src: int = 30000
    vocab_size_tgt: int = 30000
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 5000


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:

        self.input_scaled_emb = ScaledEmbedding(cfg.d_model, cfg.vocab_size_src)
        self.output_scaled_emb = ScaledEmbedding(cfg.d_model, cfg.vocab_size_tgt)
        self.input_pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.output_pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.encoder = Encoder(
            cfg.d_model,
            cfg.vocab_size_src,
            cfg.max_len,
            cfg.num_heads,
            cfg.d_ff,
            cfg.num_layers,
            cfg.dropout,
        )
        self.decoder = Decoder(
            cfg.d_model,
            cfg.vocab_size_tgt,
            cfg.max_len,
            cfg.num_heads,
            cfg.d_ff,
            cfg.num_layers,
            cfg.dropout,
        )

        self.input_emb_dropout = nn.Dropout(0.1)
        self.output_emb_dropout = nn.Dropout(0.1)

        self.projection_linear = nn.Linear(cfg.d_model, cfg.vocab_size_tgt)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.input_scaled_emb(x)  # (batch_size, seq_len, d_model)
        x = x + self.input_pos_enc(x)  # (batch_size, seq_len, d_model)
        x = self.input_emb_dropout(x)
        x = self.encoder(x, mask)  # (batch_size, seq_len, d_model)
        return x

    def decode(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.output_scaled_emb(x)  # (batch_size, seq_len, d_model)
        x = x + self.output_pos_enc(x)  # (batch_size, seq_len, d_model)
        x = self.output_emb_dropout(x)
        x = self.decoder(
            x, encoder_output, src_mask, tgt_mask
        )  # (batch_size, seq_len, d_model)
        return x

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection_linear(x)  # (batch_size, seq_len, vocab_size)
        x = torch.log_softmax(x, dim=-1)  # (batch_size, seq_len, vocab_size)
        return x
