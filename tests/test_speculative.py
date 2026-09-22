import copy
import itertools
from collections.abc import Callable, Sequence

import pytest
import torch
from conftest import ModelPair

from tinygpt.config import ModelConfig
from tinygpt.generate import IncrementalDecoder, generate
from tinygpt.model import GPT
from tinygpt.sampling import GREEDY, SamplingConfig
from tinygpt.speculative import (
    SpeculativeStats,
    speculative_generate,
    verify_greedy,
    verify_sampled,
)


@pytest.fixture(scope="module")
def target(trained_pair: ModelPair) -> GPT:
    return trained_pair.target


@pytest.fixture(scope="module")
def drafts(trained_pair: ModelPair) -> dict[str, GPT]:
    torch.manual_seed(99)
    random_cfg = ModelConfig(
        vocab_size=trained_pair.tokenizer.vocab_size, block_size=64, n_layer=1, n_head=2, n_embd=16
    )
    return {
        "identical": copy.deepcopy(trained_pair.target),
        "trained": trained_pair.draft,
        "random": GPT(random_cfg).eval(),
    }


@pytest.fixture(scope="module")
def prompt(trained_pair: ModelPair) -> list[int]:
    return trained_pair.tokenizer.encode("First Citizen:\n")


# --------------------------------------------------------------------------------------
# Greedy: speculative output must equal target-only greedy output exactly.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("draft_name", ["identical", "trained", "random"])
@pytest.mark.parametrize("k", [1, 2, 4, 7])
def test_speculative_greedy_matches_target_greedy(
    target: GPT, drafts: dict[str, GPT], prompt: list[int], draft_name: str, k: int
) -> None:
    n_new = target.config.block_size - len(prompt)
    reference = list(generate(target, prompt, n_new, GREEDY))
    stats = SpeculativeStats()
    spec = list(
        speculative_generate(target, drafts[draft_name], prompt, n_new, GREEDY, k=k, stats=stats)
    )
    assert spec == reference
    assert stats.emitted == n_new


@pytest.mark.parametrize("k", [3, 7])
def test_drafts_never_push_the_target_out_of_the_first_window(
    target: GPT,
    drafts: dict[str, GPT],
    prompt: list[int],
    k: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain decoding keeps the full context until the sequence outgrows the window.

    Speculative decoding asks the target about seq + drafts, which can be longer than seq.
    If that made the target slide its window early, it would condition on less context
    than plain decoding and the outputs could differ. Record every slide and check it
    only happens once the agreed sequence itself no longer fits.
    """
    early_slides: list[tuple[int, int]] = []

    class RecordingDecoder(IncrementalDecoder):
        def logits(self, seq: Sequence[int], n_last: int = 1) -> torch.Tensor:
            before = self.offset
            out = super().logits(seq, n_last)
            agreed = len(seq) - n_last + 1
            if self.offset != before and agreed <= self.block_size:
                early_slides.append((agreed, self.offset))
            return out

    monkeypatch.setattr("tinygpt.speculative.IncrementalDecoder", RecordingDecoder)
    block = target.config.block_size
    n_new = block - len(prompt) + 20
    reference = list(generate(target, prompt, n_new, GREEDY))
    spec = list(speculative_generate(target, drafts["trained"], prompt, n_new, GREEDY, k=k))
    assert early_slides == []
    in_window = block - len(prompt) + 1
    assert spec[:in_window] == reference[:in_window]


def test_speculative_greedy_with_repetition_penalty_matches_target(
    target: GPT, drafts: dict[str, GPT], prompt: list[int]
) -> None:
    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.3)
    reference = list(generate(target, prompt, 45, cfg))
    spec = list(speculative_generate(target, drafts["trained"], prompt, 45, cfg, k=3))
    assert spec == reference
    # The penalty should make the output less repetitive than plain greedy.
    assert reference != list(generate(target, prompt, 45, GREEDY))


def test_acceptance_rate_reflects_draft_quality(
    target: GPT, drafts: dict[str, GPT], prompt: list[int]
) -> None:
    rates = {}
    for name, draft in drafts.items():
        stats = SpeculativeStats()
        list(speculative_generate(target, draft, prompt, 45, GREEDY, k=4, stats=stats))
        rates[name] = stats.acceptance_rate
    assert rates["identical"] == 1.0
    assert rates["identical"] > rates["trained"] > rates["random"]


def test_identical_draft_emits_k_plus_one_tokens_per_round(
    target: GPT, drafts: dict[str, GPT], prompt: list[int]
) -> None:
    stats = SpeculativeStats()
    list(speculative_generate(target, drafts["identical"], prompt, 45, GREEDY, k=4, stats=stats))
    assert stats.tokens_per_round == pytest.approx(5.0)


def test_identical_draft_is_always_accepted_when_sampling(
    target: GPT, drafts: dict[str, GPT], prompt: list[int]
) -> None:
    stats = SpeculativeStats()
    gen = torch.Generator().manual_seed(0)
    cfg = SamplingConfig(temperature=1.0)
    stream = speculative_generate(
        target, drafts["identical"], prompt, 45, cfg, k=4, generator=gen, stats=stats
    )
    assert len(list(stream)) == 45
    # p == q up to float noise, so min(1, p/q) is 1 and every draft token survives.
    assert stats.acceptance_rate > 0.99


def test_stops_exactly_at_max_new_tokens(target: GPT, drafts: dict[str, GPT]) -> None:
    for n in [1, 2, 5, 13]:
        assert len(list(speculative_generate(target, drafts["trained"], [1], n, k=4))) == n


def test_rejects_bad_arguments(target: GPT, drafts: dict[str, GPT]) -> None:
    other_vocab = GPT(ModelConfig(vocab_size=7, block_size=64, n_layer=1, n_head=2, n_embd=16))
    with pytest.raises(ValueError, match="vocabulary"):
        list(speculative_generate(target, other_vocab, [1], 5))
    with pytest.raises(ValueError, match="k must be"):
        list(speculative_generate(target, drafts["trained"], [1], 5, k=0))
    with pytest.raises(ValueError, match="prompt"):
        list(speculative_generate(target, drafts["trained"], [], 5))
    with pytest.raises(ValueError, match="context window"):
        list(speculative_generate(target, drafts["trained"], [1], 5, k=40))


# --------------------------------------------------------------------------------------
# The verification rules on their own.
# --------------------------------------------------------------------------------------


def test_verify_greedy_accepts_matching_prefix() -> None:
    assert verify_greedy([4, 5, 6], [4, 5, 6, 7]) == ([4, 5, 6, 7], 3)
    assert verify_greedy([4, 5, 6], [4, 9, 6, 7]) == ([4, 9], 1)
    assert verify_greedy([4, 5, 6], [1, 5, 6, 7]) == ([1], 0)
    assert verify_greedy([], [3]) == ([3], 0)


def test_verify_sampled_accepts_when_draft_equals_target() -> None:
    probs = torch.softmax(torch.randn(4, 6, generator=torch.Generator().manual_seed(0)), dim=-1)
    gen = torch.Generator().manual_seed(1)
    for _ in range(200):
        _, n_accepted = verify_sampled([0, 1, 2], probs[:3], probs, gen)
        assert n_accepted == 3


def test_verify_sampled_rejects_tokens_the_target_rules_out() -> None:
    q = torch.tensor([[0.5, 0.5, 0.0]])
    p = torch.tensor([[0.0, 0.2, 0.8], [1.0, 0.0, 0.0]])
    gen = torch.Generator().manual_seed(0)
    for _ in range(100):
        out, n_accepted = verify_sampled([0], q, p, gen)
        assert n_accepted == 0
        # Residual max(0, p - q) is (0, 0, 0.8): the only possible replacement is 2.
        assert out == [2]


# --------------------------------------------------------------------------------------
# Speculative sampling preserves the target distribution.
#
# Toy models: first-order Markov chains over a 3-token vocabulary. The target's
# next-token distribution is row x_{t-1} of P, the draft's is row x_{t-1} of Q.
# Running speculative sampling for a few tokens, the joint distribution of the
# output must be the target chain's, P[x0, x1] * P[x1, x2] * P[x2, x3], and not
# anything involving Q.
# --------------------------------------------------------------------------------------

P = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
Q = torch.tensor([[0.2, 0.2, 0.6], [0.5, 0.4, 0.1], [0.3, 0.1, 0.6]])
LENGTH = 3
START = 0
N_SAMPLES = 20_000


Verifier = Callable[
    [Sequence[int], torch.Tensor, torch.Tensor, torch.Generator | None], tuple[list[int], int]
]


def speculative_markov_sample(
    k: int, gen: torch.Generator, verify: Verifier = verify_sampled
) -> tuple[int, ...]:
    """The same round structure as ``speculative_generate``, with toy models."""
    seq = [START]
    while len(seq) - 1 < LENGTH:
        n_draft = min(k, LENGTH - (len(seq) - 1) - 1)
        drafted: list[int] = []
        draft_rows = []
        for _ in range(n_draft):
            q = Q[(seq + drafted)[-1]]
            drafted.append(int(torch.multinomial(q, 1, generator=gen)))
            draft_rows.append(q)
        target_rows = torch.stack([P[(seq + drafted[:i])[-1]] for i in range(n_draft + 1)])
        draft_probs = torch.stack(draft_rows) if draft_rows else torch.zeros(0, 3)
        new_tokens, _ = verify(drafted, draft_probs, target_rows, gen)
        seq.extend(new_tokens)
    return tuple(seq[1 : LENGTH + 1])


def chain_joint(matrix: torch.Tensor) -> dict[tuple[int, ...], float]:
    joint = {}
    for path in itertools.product(range(3), repeat=LENGTH):
        prob, prev = 1.0, START
        for token in path:
            prob *= float(matrix[prev, token])
            prev = token
        joint[path] = prob
    return joint


def test_toy_draft_really_differs_from_target() -> None:
    """If the test below accidentally measured the draft, it would fail by a wide margin."""
    target, draft = chain_joint(P), chain_joint(Q)
    tv = 0.5 * sum(abs(target[x] - draft[x]) for x in target)
    assert tv > 0.3


def toy_statistics(k: int, seed: int, verify: Verifier = verify_sampled) -> tuple[float, float]:
    """Chi-square statistic and total variation of speculative samples against the target."""
    gen = torch.Generator().manual_seed(seed)
    counts: dict[tuple[int, ...], int] = dict.fromkeys(chain_joint(P), 0)
    for _ in range(N_SAMPLES):
        counts[speculative_markov_sample(k, gen, verify)] += 1
    expected = {x: pr * N_SAMPLES for x, pr in chain_joint(P).items()}
    chi2 = sum((counts[x] - expected[x]) ** 2 / expected[x] for x in expected)
    tv = 0.5 * sum(abs(counts[x] - expected[x]) for x in expected) / N_SAMPLES
    return chi2, tv


@pytest.mark.parametrize("k", [1, 2, 3])
def test_speculative_sampling_preserves_target_distribution(k: int) -> None:
    chi2, tv = toy_statistics(k, seed=1234 + k)
    assert chi2 < chi_square_critical(3**LENGTH - 1), f"chi2={chi2:.1f}"
    assert tv < 0.02, f"tv={tv:.4f}"


def test_distribution_check_catches_a_subtly_wrong_rule() -> None:
    """A classic mistake: on rejection, resample from p instead of the residual (p - q)+.

    That over-weights tokens the draft already proposes often. The same check must fail.
    """

    def resample_from_p(
        tokens: Sequence[int], q: torch.Tensor, p: torch.Tensor, gen: torch.Generator | None
    ) -> tuple[list[int], int]:
        out: list[int] = []
        for i, token in enumerate(tokens):
            if float(torch.rand((), generator=gen)) * float(q[i, token]) < float(p[i, token]):
                out.append(token)
                continue
            out.append(int(torch.multinomial(p[i], 1, generator=gen)))
            return out, i
        out.append(int(torch.multinomial(p[len(tokens)], 1, generator=gen)))
        return out, len(tokens)

    chi2, tv = toy_statistics(k=2, seed=7, verify=resample_from_p)
    assert chi2 > chi_square_critical(3**LENGTH - 1)
    assert tv > 0.05


def test_speculative_sampling_with_real_models_follows_the_target(trained_pair: ModelPair) -> None:
    """End to end through speculative_generate: the first sampled token follows the
    target's next-token distribution, not the draft's."""
    target, draft, tok = trained_pair.target, trained_pair.draft, trained_pair.tokenizer
    prompt = tok.encode("You are ")
    with torch.no_grad():
        p = torch.softmax(target(torch.tensor([prompt]))[0, -1], dim=-1)
        q = torch.softmax(draft(torch.tensor([prompt]))[0, -1], dim=-1)
    # The draft must disagree enough for the chi-square check against q below to mean
    # something. The exact gap depends on platform float differences during training
    # (about 0.14 on Linux CI, higher on macOS), so keep the floor loose.
    assert 0.5 * (p - q).abs().sum() > 0.1

    gen = torch.Generator().manual_seed(0)
    cfg = SamplingConfig(temperature=1.0)
    n = 4000
    counts = torch.zeros(tok.vocab_size)
    for _ in range(n):
        stream = speculative_generate(target, draft, prompt, 2, cfg, k=1, generator=gen)
        counts[next(stream)] += 1

    chi2, df = pooled_chi_square(counts, p * n)
    assert chi2 < chi_square_critical(df), f"chi2={chi2:.1f} df={df}"
    chi2_vs_draft, df_draft = pooled_chi_square(counts, q * n)
    assert chi2_vs_draft > chi_square_critical(df_draft)


def pooled_chi_square(observed: torch.Tensor, expected: torch.Tensor) -> tuple[float, int]:
    """Pearson chi-square, pooling cells with expected count < 20 into one bin."""
    big = expected >= 20
    obs = [*observed[big].tolist(), float(observed[~big].sum())]
    exp = [*expected[big].tolist(), float(expected[~big].sum())]
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp, strict=True) if e > 0)
    return stat, len(obs) - 1


def chi_square_critical(df: int, z: float = 3.09) -> float:
    """Upper 0.1% point of chi-square via the Wilson-Hilferty approximation."""
    a = 2.0 / (9.0 * df)
    return float(df * (1.0 - a + z * a**0.5) ** 3)
