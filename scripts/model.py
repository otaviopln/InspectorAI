"""Inspector AI — Sequence-to-Sequence Transformer (from scratch).

A custom PyTorch Transformer that learns to translate raw user prompts into
either:
  1. strategic clarification questions  (when the prompt is dubious / vague), or
  2. R.C.F.O formatted prompts         (when the prompt is clear).

Every component below is coded from torch primitives — no HuggingFace,
no sentencepiece, no third-party NLP libraries:

  - PositionalEncoding      : sinusoidal position injection
  - MultiHeadAttention      : scaled dot-product attention with multiple heads
  - EncoderLayer / Encoder  : self-attention + feed-forward stack
  - DecoderLayer / Decoder  : masked self-attention + cross-attention + FFN
  - InspectorTransformer    : full encoder-decoder Seq2Seq model
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Positional Encoding (sinusoidal)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Injects absolute position information into token embeddings.

    Uses sine/cosine functions of different frequencies so the model can
    attend to relative and absolute positions without recurrence.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention (self-attention + cross-attention)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over multiple representation subspaces.

    Math (per head h):
        Q_h = X W_q^h,  K_h = X W_k^h,  V_h = X W_v^h
        scores_h = Q_h K_h^T / sqrt(d_k)
        attn_h   = softmax(scores_h + mask)
        out_h    = attn_h V_h

    All heads run in parallel; outputs are concatenated and projected back.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = q.shape
        k_len = k.size(1)

        # Project and split into heads: (batch, num_heads, len, head_dim)
        Q = (
            self.q_proj(q)
            .view(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(k)
            .view(batch_size, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(v)
            .view(batch_size, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product scores: (batch, num_heads, q_len, k_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask == 0 positions are set to -inf before softmax
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)  # (batch, num_heads, q_len, head_dim)

        # Merge heads back: (batch, q_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 3. Encoder
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """One Transformer encoder block: self-attention + feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention: queries, keys, values all come from encoder input
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))

        # Position-wise feed-forward
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class Encoder(nn.Module):
    """Stack of N identical encoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Scale embeddings by sqrt(d_model) per original Transformer paper
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ---------------------------------------------------------------------------
# 4. Decoder
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """One Transformer decoder block:

    1. Masked multi-head self-attention (cannot see future tokens)
    2. Multi-head cross-attention (queries from decoder, keys/values from encoder)
    3. Position-wise feed-forward
    Each sub-layer has a residual connection + layer norm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked self-attention (decoder only attends to past tokens)
        attn1 = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # 2. Cross-attention: decoder queries attend to encoder keys/values
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # 3. Feed-forward
        ff = self.ffn(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class Decoder(nn.Module):
    """Stack of N identical decoder layers followed by output projection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)

        return self.output_proj(x)  # logits over vocabulary


# ---------------------------------------------------------------------------
# 5. Full Seq2Seq Transformer
# ---------------------------------------------------------------------------
class InspectorTransformer(nn.Module):
    """Encoder-decoder Transformer for prompt -> questions / R.C.F.O translation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.encoder = Encoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)

    # -- Mask helpers ---------------------------------------------------------
    @staticmethod
    def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Mask padded positions: (batch, 1, 1, seq_len)."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Lower-triangular mask so position i cannot attend to j > i.

        Returns shape (1, 1, seq_len, seq_len) so it broadcasts over batch
        and heads.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Combine padding mask + causal look-ahead mask for the target."""
        pad_mask = self.create_padding_mask(tgt, self.pad_idx)
        causal_mask = self.create_causal_mask(tgt.size(1), tgt.device)
        return pad_mask & causal_mask  # logical AND

    # -- Forward --------------------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.create_padding_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, src_mask, tgt_mask)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 300
    batch, src_len, tgt_len = 2, 10, 8

    model = InspectorTransformer(
        vocab_size=vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=256
    )
    model.eval()

    src = torch.randint(0, vocab_size, (batch, src_len))
    tgt = torch.randint(0, vocab_size, (batch, tgt_len))

    with torch.no_grad():
        logits = model(src, tgt)

    print(f"src shape : {src.shape}")
    print(f"tgt shape : {tgt.shape}")
    print(f"logits shape: {logits.shape}  (batch, tgt_len, vocab_size)")
    print("InspectorTransformer forward pass OK.")
