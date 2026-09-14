"""Key/value cache for incremental decoding.

During generation every new token attends to the keys and values of all the
tokens before it. Those don't change once computed (the mask is causal and
positions are fixed by RoPE at write time), so we store them and only run the
new tokens through the model.

Storage is preallocated to ``max_len`` so appending is a slice write, and
rolling back (which speculative decoding needs after a rejected draft) is just
moving ``pos`` backwards.
"""

from __future__ import annotations

import torch
from torch import Tensor


class KVCache:
    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        n_head: int,
        max_len: int,
        head_dim: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layer, batch_size, n_head, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    @property
    def max_len(self) -> int:
        return self.k.size(3)

    def __len__(self) -> int:
        return self.pos

    def update(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write ``k``/``v`` of shape ``(B, H, T, D)`` at the current position for one layer.

        Returns views over everything cached so far for that layer, including the
        new entries. ``pos`` is not advanced here because every layer writes to the
        same positions; the model calls :meth:`advance` once after the last layer.
        """
        end = self.pos + k.size(2)
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: need {end} positions, have {self.max_len}")
        self.k[layer, :, :, self.pos : end] = k
        self.v[layer, :, :, self.pos : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        self.pos += n

    def crop(self, length: int) -> None:
        """Forget everything from position ``length`` onwards."""
        if not 0 <= length <= self.pos:
            raise ValueError(f"cannot crop cache of length {self.pos} to {length}")
        self.pos = length

    def reset(self) -> None:
        self.pos = 0
