"""Autoregressive generation as a streaming iterator of token ids."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from tinygpt.model import GPT
from tinygpt.sampling import GREEDY, SamplingConfig, sample_token


def generate(
    model: GPT,
    prompt: Sequence[int],
    max_new_tokens: int,
    sampling: SamplingConfig = GREEDY,
    *,
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    """Yield up to ``max_new_tokens`` new token ids, one at a time.

    Every step re-runs the model over the whole context (the last ``block_size``
    tokens), which is simple and quadratic in the output length.
    """
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    model.eval()
    device = next(model.parameters()).device
    block_size = model.config.block_size
    seq = list(prompt)
    for _ in range(max_new_tokens):
        # Keep inference mode scoped to the step, not held open across the yield.
        with torch.inference_mode():
            idx = torch.tensor([seq[-block_size:]], dtype=torch.long, device=device)
            logits = model(idx)[0, -1]
        token = sample_token(logits, sampling, seq, generator)
        seq.append(token)
        yield token
