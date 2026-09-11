"""Turning logits into a next token.

The processing order is: repetition penalty, temperature, top-k, top-p. The
same function produces the distribution for plain sampling and for both sides
of speculative sampling, so the speculative path samples from exactly the
distribution the plain path would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingConfig:
    """``temperature == 0`` means greedy decoding (argmax after the repetition penalty)."""

    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


GREEDY = SamplingConfig(temperature=0.0)


def apply_repetition_penalty(logits: Tensor, history: Sequence[int], penalty: float) -> Tensor:
    """CTRL-style penalty (Keskar et al., 2019) on every token id already in ``history``.

    Positive logits are divided by ``penalty`` and negative ones multiplied, so a
    penalty above 1 always makes a repeat less likely.
    """
    if penalty == 1.0 or not history:
        return logits
    ids = torch.tensor(sorted(set(history)), dtype=torch.long, device=logits.device)
    seen = logits[..., ids]
    penalised = torch.where(seen > 0, seen / penalty, seen * penalty)
    out = logits.clone()
    out[..., ids] = penalised
    return out


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the ``k`` largest logits and set the rest to ``-inf``. Ties at the cut-off are kept."""
    k = min(k, logits.size(-1))
    kth_largest = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth_largest, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering (Holtzman et al., 2019).

    Keep the smallest set of most-likely tokens whose probability mass reaches
    ``p``. The most likely token always survives.
    """
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs
    drop_sorted = mass_before >= p
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def process_logits(logits: Tensor, config: SamplingConfig, history: Sequence[int] = ()) -> Tensor:
    """Apply every configured transform to a ``(..., vocab)`` logits tensor."""
    logits = apply_repetition_penalty(logits.float(), history, config.repetition_penalty)
    if config.greedy:
        return logits
    logits = logits / config.temperature
    if config.top_k is not None:
        logits = top_k_filter(logits, config.top_k)
    if config.top_p is not None:
        logits = top_p_filter(logits, config.top_p)
    return logits


def next_token_probs(logits: Tensor, config: SamplingConfig, history: Sequence[int] = ()) -> Tensor:
    """Distribution over the next token. Greedy decoding is a one-hot distribution."""
    processed = process_logits(logits, config, history)
    if config.greedy:
        return torch.nn.functional.one_hot(
            processed.argmax(dim=-1), num_classes=processed.size(-1)
        ).float()
    return torch.softmax(processed, dim=-1)


def sample_token(
    logits: Tensor,
    config: SamplingConfig,
    history: Sequence[int] = (),
    generator: torch.Generator | None = None,
) -> int:
    """Pick one token from a single ``(vocab,)`` row of logits."""
    logits = logits.detach().cpu()
    if config.greedy:
        return int(process_logits(logits, config, history).argmax())
    probs = next_token_probs(logits, config, history)
    return int(torch.multinomial(probs, 1, generator=generator))
