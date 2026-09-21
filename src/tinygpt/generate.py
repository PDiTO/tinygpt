"""Autoregressive generation as a streaming iterator of token ids."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch import Tensor

from tinygpt.model import GPT
from tinygpt.sampling import GREEDY, SamplingConfig, sample_token


class IncrementalDecoder:
    """Runs a model over a growing token sequence.

    With ``use_cache=True`` only the tokens the KV cache hasn't seen are fed to the
    model, so each decoding step costs one token's worth of compute instead of the
    whole context. Without the cache every call re-runs the full window; that path
    exists as the reference the cached one is tested against, and as the baseline
    for benchmarks.

    Invariant: the cache holds keys/values for ``seq[offset : offset + cache.pos]``.
    Callers that drop tokens from the end of the sequence (speculative decoding
    does, when a draft is rejected) must call :meth:`rollback`.

    Once the sequence outgrows the model's context window the cached path slides
    forward by half a window and re-fills the cache. The uncached path slides by one
    token per step instead, so the two only agree token-for-token while everything
    fits in ``block_size``.
    """

    def __init__(self, model: GPT, *, use_cache: bool = True) -> None:
        self.model = model
        self.block_size = model.config.block_size
        self.device = next(model.parameters()).device
        self.cache = model.new_cache(batch_size=1) if use_cache else None
        self.offset = 0
        self.forward_calls = 0

    def logits(self, seq: Sequence[int], n_last: int = 1) -> Tensor:
        """Logits ``(n_last, vocab)`` predicting the token after each of the last ``n_last``."""
        if not 1 <= n_last <= len(seq):
            raise ValueError(f"n_last must be in [1, {len(seq)}], got {n_last}")
        self.forward_calls += 1

        if self.cache is None:
            window = seq[-self.block_size :]
            if n_last > len(window):
                raise ValueError("n_last exceeds the context window")
            out: Tensor = self.model(self._tensor(window))
            return out[0, -n_last:]

        if n_last > self.block_size // 2:
            raise ValueError(f"n_last must be at most half the context ({self.block_size // 2})")
        if len(seq) - self.offset > self.block_size:
            # Out of room: keep the most recent half window and re-fill the cache.
            self.offset = len(seq) - self.block_size // 2
            self.cache.reset()

        cached_until = self.offset + self.cache.pos
        if cached_until > len(seq) - n_last:
            raise RuntimeError(
                f"cache already covers position {cached_until - 1}, but logits were requested "
                f"from position {len(seq) - n_last}; call rollback() first"
            )
        new_tokens = seq[cached_until:]
        out = self.model(self._tensor(new_tokens), cache=self.cache)
        return out[0, -n_last:]

    def rollback(self, length: int) -> None:
        """Drop cached positions at or beyond absolute position ``length``."""
        if self.cache is None or self.offset + self.cache.pos <= length:
            return
        if length >= self.offset:
            self.cache.crop(length - self.offset)
        else:
            # Rolled back past the start of the current window: start a fresh one.
            self.cache.reset()
            self.offset = max(0, length - self.block_size // 2)

    def _tensor(self, tokens: Sequence[int]) -> Tensor:
        return torch.tensor([list(tokens)], dtype=torch.long, device=self.device)


def generate(
    model: GPT,
    prompt: Sequence[int],
    max_new_tokens: int,
    sampling: SamplingConfig = GREEDY,
    *,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    """Yield up to ``max_new_tokens`` new token ids, one at a time.

    The model is switched to eval mode. Pass a seeded ``generator`` for
    reproducible sampling.
    """
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    model.eval()
    decoder = IncrementalDecoder(model, use_cache=use_cache)
    seq = list(prompt)
    for _ in range(max_new_tokens):
        # Keep inference mode scoped to the step rather than held open across the yield,
        # otherwise it would leak into whatever the caller does between tokens.
        with torch.inference_mode():
            logits = decoder.logits(seq)[-1]
        token = sample_token(logits, sampling, seq, generator)
        seq.append(token)
        yield token
