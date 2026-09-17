"""Wall-clock decoding benchmarks.

Every case generates the same number of tokens from the same prompt, at batch
size 1, and reports the median of several timed runs after a warm-up run.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

from tinygpt.generate import generate
from tinygpt.model import GPT
from tinygpt.sampling import GREEDY, SamplingConfig
from tinygpt.speculative import SpeculativeStats, speculative_generate
from tinygpt.utils import synchronize


@dataclass(frozen=True)
class BenchResult:
    name: str
    mode: str
    tokens: int
    seconds: float
    acceptance_rate: float | None = None
    tokens_per_round: float | None = None

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds


# A case takes a seed, generates tokens, and returns speculative stats if it has any.
Case = Callable[[int], tuple[list[int], SpeculativeStats | None]]


def time_case(
    case: Case, n_tokens: int, repeats: int, device: torch.device
) -> tuple[float, SpeculativeStats | None]:
    """Median seconds over ``repeats`` runs, after one untimed warm-up."""
    case(-1)
    times: list[float] = []
    merged: SpeculativeStats | None = None
    for seed in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        tokens, stats = case(seed)
        synchronize(device)
        times.append(time.perf_counter() - start)
        if len(tokens) != n_tokens:
            raise RuntimeError(f"expected {n_tokens} tokens, got {len(tokens)}")
        if stats is not None:
            merged = merged or SpeculativeStats()
            merged.rounds += stats.rounds
            merged.proposed += stats.proposed
            merged.accepted += stats.accepted
            merged.emitted += stats.emitted
    return statistics.median(times), merged


def plain_case(
    model: GPT, prompt: Sequence[int], n: int, sampling: SamplingConfig, *, cache: bool
) -> Case:
    def run(seed: int) -> tuple[list[int], SpeculativeStats | None]:
        gen = torch.Generator().manual_seed(seed)
        return list(generate(model, prompt, n, sampling, use_cache=cache, generator=gen)), None

    return run


def speculative_case(
    target: GPT, draft: GPT, prompt: Sequence[int], n: int, sampling: SamplingConfig, k: int
) -> Case:
    def run(seed: int) -> tuple[list[int], SpeculativeStats | None]:
        gen = torch.Generator().manual_seed(seed)
        stats = SpeculativeStats()
        tokens = list(
            speculative_generate(
                target, draft, prompt, n, sampling, k=k, generator=gen, stats=stats
            )
        )
        return tokens, stats

    return run


def run_benchmarks(
    target: GPT,
    draft: GPT | None,
    prompt: Sequence[int],
    n_tokens: int,
    *,
    repeats: int = 3,
    spec_ks: Iterable[int] = (4,),
    temperature: float = 0.8,
    device: torch.device | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[BenchResult]:
    device = device or next(target.parameters()).device
    modes = {
        "greedy": GREEDY,
        f"sampled T={temperature:g}": SamplingConfig(temperature=temperature),
    }
    results: list[BenchResult] = []
    for mode, sampling in modes.items():
        cases: list[tuple[str, Case]] = [
            ("target, no cache", plain_case(target, prompt, n_tokens, sampling, cache=False)),
            ("target, KV cache", plain_case(target, prompt, n_tokens, sampling, cache=True)),
        ]
        if draft is not None:
            cases.append(
                ("draft, KV cache", plain_case(draft, prompt, n_tokens, sampling, cache=True))
            )
            for k in spec_ks:
                cases.append(
                    (
                        f"speculative, k={k}",
                        speculative_case(target, draft, prompt, n_tokens, sampling, k),
                    )
                )
        for name, case in cases:
            if progress:
                progress(f"{mode}: {name}")
            seconds, stats = time_case(case, n_tokens, repeats, device)
            results.append(
                BenchResult(
                    name=name,
                    mode=mode,
                    tokens=n_tokens,
                    seconds=seconds,
                    acceptance_rate=stats.acceptance_rate if stats else None,
                    tokens_per_round=stats.tokens_per_round if stats else None,
                )
            )
    return results


def format_table(results: Sequence[BenchResult]) -> str:
    """Markdown table. Speed-up is relative to the cached target in the same mode."""
    baseline = {r.mode: r.tokens_per_second for r in results if r.name == "target, KV cache"}
    lines = [
        "| mode | method | tokens/s | vs KV cache | acceptance | tokens/target pass |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        speedup = r.tokens_per_second / baseline[r.mode] if r.mode in baseline else float("nan")
        accept = f"{r.acceptance_rate:.1%}" if r.acceptance_rate is not None else ""
        per_round = f"{r.tokens_per_round:.2f}" if r.tokens_per_round is not None else ""
        lines.append(
            f"| {r.mode} | {r.name} | {r.tokens_per_second:,.0f} | {speedup:.2f}x "
            f"| {accept} | {per_round} |"
        )
    return "\n".join(lines)
