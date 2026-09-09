"""Model hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Shape of a :class:`tinygpt.model.GPT`.

    ``mlp_hidden`` defaults to roughly ``8/3 * n_embd`` rounded up to a multiple of 32,
    which keeps the SwiGLU block at about the same parameter count as a 4x GELU MLP.
    """

    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_hidden: int | None = None
    dropout: float = 0.0
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def hidden_dim(self) -> int:
        if self.mlp_hidden is not None:
            return self.mlp_hidden
        hidden = int(8 * self.n_embd / 3)
        return 32 * ((hidden + 31) // 32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        return cls(**data)
