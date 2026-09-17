import pytest
from conftest import ModelPair

from tinygpt.bench import BenchResult, format_table, run_benchmarks


def test_run_benchmarks_covers_every_case(trained_pair: ModelPair) -> None:
    prompt = trained_pair.tokenizer.encode("All:")
    results = run_benchmarks(
        trained_pair.target, trained_pair.draft, prompt, 12, repeats=1, spec_ks=(2,)
    )
    names = [(r.mode, r.name) for r in results]
    assert len(names) == 8
    assert ("greedy", "speculative, k=2") in names
    for r in results:
        assert r.tokens == 12
        assert r.seconds > 0
        is_spec = r.name.startswith("speculative")
        assert (r.acceptance_rate is not None) == is_spec


def test_format_table_reports_speedup_against_the_cached_target() -> None:
    results = [
        BenchResult("target, no cache", "greedy", 100, 2.0),
        BenchResult("target, KV cache", "greedy", 100, 1.0),
        BenchResult(
            "speculative, k=4", "greedy", 100, 0.5, acceptance_rate=0.5, tokens_per_round=3
        ),
    ]
    table = format_table(results).splitlines()
    assert table[2].endswith("| 50 | 0.50x |  |  |")
    assert "| 100 | 1.00x |" in table[3]
    assert "| 200 | 2.00x | 50.0% | 3.00 |" in table[4]


def test_tokens_per_second() -> None:
    assert BenchResult("x", "greedy", 30, 1.5).tokens_per_second == pytest.approx(20.0)
