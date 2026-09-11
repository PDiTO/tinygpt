import math

import pytest
import torch

from tinygpt.config import ModelConfig
from tinygpt.generate import generate
from tinygpt.model import GPT
from tinygpt.sampling import (
    GREEDY,
    SamplingConfig,
    apply_repetition_penalty,
    next_token_probs,
    process_logits,
    sample_token,
    top_k_filter,
    top_p_filter,
)

NEG_INF = float("-inf")


def kept(logits: torch.Tensor) -> list[int]:
    return torch.nonzero(torch.isfinite(logits)).flatten().tolist()


def test_top_k_keeps_the_k_largest() -> None:
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    out = top_k_filter(logits, 2)
    assert kept(out) == [1, 4]
    # Survivors are untouched.
    assert out[1] == 5.0
    assert out[4] == 4.0


def test_top_k_keeps_ties_at_the_boundary() -> None:
    out = top_k_filter(torch.tensor([1.0, 3.0, 3.0, 0.0]), 1)
    assert kept(out) == [1, 2]


def test_top_k_larger_than_vocab_is_a_no_op() -> None:
    logits = torch.tensor([0.3, -1.0, 2.0])
    torch.testing.assert_close(top_k_filter(logits, 10), logits)


def test_top_k_works_per_row() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    out = top_k_filter(logits, 1)
    assert torch.isfinite(out).tolist() == [[False, False, True], [True, False, False]]


# Probabilities 0.5, 0.2, 0.15, 0.1, 0.05 deliberately stored out of order.
NUCLEUS_PROBS = torch.tensor([0.1, 0.5, 0.05, 0.2, 0.15])
NUCLEUS_LOGITS = NUCLEUS_PROBS.log()


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.01, [1]),  # the most likely token always survives
        (0.5, [1]),  # 0.5 alone reaches p
        (0.6, [1, 3]),  # 0.5 + 0.2
        (0.8, [1, 3, 4]),  # 0.5 + 0.2 + 0.15
        (0.9, [0, 1, 3, 4]),
        (1.0, [0, 1, 2, 3, 4]),
    ],
)
def test_top_p_keeps_smallest_set_reaching_p(p: float, expected: list[int]) -> None:
    assert kept(top_p_filter(NUCLEUS_LOGITS, p)) == expected


def test_top_p_renormalised_distribution() -> None:
    probs = torch.softmax(top_p_filter(NUCLEUS_LOGITS, 0.6), dim=-1)
    expected = torch.zeros(5)
    expected[1] = 0.5 / 0.7
    expected[3] = 0.2 / 0.7
    torch.testing.assert_close(probs, expected)


def test_top_p_works_per_row() -> None:
    logits = torch.stack([NUCLEUS_LOGITS, NUCLEUS_LOGITS.roll(1)])
    out = top_p_filter(logits, 0.6)
    assert kept(out[0]) == [1, 3]
    assert kept(out[1]) == [2, 4]


def test_repetition_penalty_pushes_seen_tokens_down() -> None:
    logits = torch.tensor([2.0, -2.0, 1.0, 0.5])
    out = apply_repetition_penalty(logits, [0, 1, 1], penalty=2.0)
    torch.testing.assert_close(out, torch.tensor([1.0, -4.0, 1.0, 0.5]))
    # Input is not modified in place.
    assert logits[0] == 2.0


def test_repetition_penalty_of_one_is_identity() -> None:
    logits = torch.randn(10)
    assert apply_repetition_penalty(logits, [1, 2, 3], 1.0) is logits


def test_temperature_scales_logits() -> None:
    logits = torch.tensor([1.0, 2.0, 0.5])
    probs = next_token_probs(logits, SamplingConfig(temperature=2.0))
    torch.testing.assert_close(probs, torch.softmax(logits / 2.0, dim=-1))


def test_greedy_probs_are_one_hot_at_argmax() -> None:
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    torch.testing.assert_close(next_token_probs(logits, GREEDY), torch.tensor([0.0, 1, 0, 0]))
    assert sample_token(logits, GREEDY) == 1


def test_greedy_respects_repetition_penalty() -> None:
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.5)
    assert sample_token(logits, cfg, history=[1]) == 3


@pytest.mark.parametrize("seed", range(5))
def test_temperature_to_zero_matches_greedy(seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(50, generator=gen)
    cold = SamplingConfig(temperature=1e-4)
    samples = {sample_token(logits, cold, generator=gen) for _ in range(50)}
    assert samples == {sample_token(logits, GREEDY)}
    assert samples == {int(logits.argmax())}


def test_samples_never_leave_the_top_k() -> None:
    logits = torch.tensor([0.0, 0.1, 0.2, 5.0, 4.0, 0.3])
    cfg = SamplingConfig(temperature=5.0, top_k=2)
    gen = torch.Generator().manual_seed(0)
    assert {sample_token(logits, cfg, generator=gen) for _ in range(500)} == {3, 4}


def test_sampling_matches_softmax_frequencies() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    gen = torch.Generator().manual_seed(0)
    n = 20_000
    counts = torch.zeros(3)
    for _ in range(n):
        counts[sample_token(logits, SamplingConfig(), generator=gen)] += 1
    expected = torch.softmax(logits, dim=-1)
    tv = 0.5 * (counts / n - expected).abs().sum()
    assert tv < 0.015


def test_process_logits_order_penalty_then_temperature() -> None:
    logits = torch.tensor([4.0, 1.0])
    cfg = SamplingConfig(temperature=2.0, repetition_penalty=2.0)
    torch.testing.assert_close(process_logits(logits, cfg, [0]), torch.tensor([1.0, 0.5]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -1.0},
        {"top_k": 0},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0},
    ],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        SamplingConfig(**kwargs)  # type: ignore[arg-type]


def tiny_model(block_size: int = 16) -> GPT:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=11, block_size=block_size, n_layer=2, n_head=2, n_embd=16)
    return GPT(cfg).eval()


def test_generate_streams_the_requested_number_of_tokens() -> None:
    stream = generate(tiny_model(), [1, 2, 3], 5)
    first = next(stream)
    assert isinstance(first, int)
    rest = list(stream)
    assert len(rest) == 4
    assert all(0 <= t < 11 for t in [first, *rest])


def test_generate_is_reproducible_with_a_seeded_generator() -> None:
    model = tiny_model()
    cfg = SamplingConfig(temperature=1.0, top_p=0.9)
    a = list(generate(model, [1], 20, cfg, generator=torch.Generator().manual_seed(3)))
    b = list(generate(model, [1], 20, cfg, generator=torch.Generator().manual_seed(3)))
    assert a == b


def test_generate_runs_past_the_context_window() -> None:
    model = tiny_model(block_size=8)
    assert len(list(generate(model, [1, 2], 30))) == 30


def test_generate_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        list(generate(tiny_model(), [], 3))


def test_uniform_logits_give_uniform_probs() -> None:
    probs = next_token_probs(torch.zeros(4), SamplingConfig(temperature=0.7))
    torch.testing.assert_close(probs, torch.full((4,), 0.25))
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-6)
