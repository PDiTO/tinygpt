"""A decoder-only transformer, written out by hand.

Pre-norm blocks with RMSNorm, rotary position embeddings, causal multi-head
self-attention and a SwiGLU MLP. The output projection shares its weights with
the token embedding.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinygpt.config import ModelConfig
from tinygpt.kv_cache import KVCache


class RMSNorm(nn.Module):
    """Scale by the reciprocal root-mean-square of the features. No mean subtraction, no bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate feature pairs ``(x[i], x[i + d/2])`` by position-dependent angles.

    ``x`` is ``(..., T, D)``; ``cos`` and ``sin`` are ``(T, D/2)``.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embedding (Su et al., 2021).

    Rotating queries and keys by an angle proportional to their absolute position
    makes ``q_m . k_n`` depend only on ``m - n``, so attention sees relative position
    without any learned position table.
    """

    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, max_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
        # Derived from the config, so keep them out of the state dict.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @property
    def max_len(self) -> int:
        return self.cos.size(0)

    def forward(self, x: Tensor, start: int = 0) -> Tensor:
        """Rotate ``x`` of shape ``(B, H, T, D)`` as if it starts at position ``start``."""
        end = start + x.size(-2)
        if end > self.max_len:
            raise ValueError(f"position {end - 1} is beyond the rotary table ({self.max_len})")
        return apply_rotary(x, self.cos[start:end], self.sin[start:end])


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        mask: Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        """Attend from the ``T`` new positions in ``x`` (``(B, T, C)``) to every visible key.

        Without a cache the keys are just the new positions. With a cache, the new
        positions start at ``cache.pos`` and the keys also include everything cached
        before them. ``mask`` is boolean ``(T, S)``, True where attending is allowed,
        with ``S`` the total number of keys.
        """
        batch, seq_len, channels = x.shape
        start = cache.pos if cache is not None else 0
        q, k, v = self.qkv(x).split(channels, dim=-1)
        # (B, T, C) -> (B, H, T, D)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Rotate by absolute position before caching, so cached keys never need touching again.
        q = rope(q, start)
        k = rope(k, start)
        if cache is not None:
            k, v = cache.update(layer, k, v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        weights = self.attn_dropout(weights)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.proj(out))  # type: ignore[no-any-return]


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))`` from Shazeer (2020)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))  # type: ignore[no-any-return]


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        mask: Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope, mask, cache, layer)
        return x + self.mlp(self.mlp_norm(x))  # type: ignore[no-any-return]


class GPT(nn.Module):
    """Decoder-only language model. ``forward`` maps token ids to next-token logits."""

    causal_mask: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied input/output embeddings

        self.rope = RotaryEmbedding(config.head_dim, config.block_size, config.rope_base)
        mask = torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: shrink the residual projections so the residual stream's
        # variance doesn't grow with depth at init.
        resid_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(param, mean=0.0, std=resid_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Parameter count, with the tied embedding counted once."""
        return sum(p.numel() for p in self.parameters())

    def new_cache(self, batch_size: int = 1) -> KVCache:
        """An empty KV cache sized for this model's full context window."""
        param = self.tok_emb.weight
        cfg = self.config
        return KVCache(
            cfg.n_layer,
            batch_size,
            cfg.n_head,
            cfg.block_size,
            cfg.head_dim,
            device=param.device,
            dtype=param.dtype,
        )

    def forward(self, idx: Tensor, cache: KVCache | None = None) -> Tensor:
        """Map token ids ``(B, T)`` to next-token logits ``(B, T, vocab_size)``.

        With a ``cache``, ``idx`` holds only the tokens the cache hasn't seen yet;
        they are placed at positions ``cache.pos`` onwards and the cache advances.
        """
        _, seq_len = idx.shape
        start = cache.pos if cache is not None else 0
        end = start + seq_len
        if end > self.config.block_size:
            raise ValueError(f"sequence length {end} exceeds block_size {self.config.block_size}")
        # Query i sits at absolute position start + i and may see keys 0..start + i.
        mask = self.causal_mask[start:end, :end]
        x = self.dropout(self.tok_emb(idx))
        for layer, block in enumerate(self.blocks):
            x = block(x, self.rope, mask, cache, layer)
        if cache is not None:
            cache.advance(seq_len)
        return self.lm_head(self.norm(x))  # type: ignore[no-any-return]
