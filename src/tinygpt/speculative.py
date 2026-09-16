"""Speculative decoding (Leviathan et al., 2023; Chen et al., 2023).

A cheap draft model proposes ``k`` tokens one at a time. The target model then
scores all of them in a single forward pass and we keep the longest prefix it
agrees with, plus one token of its own. Each round costs one target forward
pass and yields between 1 and ``k + 1`` tokens.

Sampling mode uses the accept/reject rule from the papers, which makes the
output distributed exactly as if the target had sampled on its own:

* accept draft token ``x`` (drawn from the draft's ``q``) with probability
  ``min(1, p(x) / q(x))``, where ``p`` is the target's distribution;
* on the first rejection, draw the replacement from the residual
  ``max(0, p - q) / sum(max(0, p - q))`` and stop;
* if all ``k`` are accepted, draw a bonus token from the target's next ``p``.

Greedy mode is the same rule with one-hot distributions, which reduces to
"accept while the draft's argmax equals the target's argmax". It produces the
same tokens as greedy decoding with the target alone.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from tinygpt.generate import IncrementalDecoder
from tinygpt.model import GPT
from tinygpt.sampling import GREEDY, SamplingConfig, next_token_probs, process_logits


@dataclass
class SpeculativeStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    emitted: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target kept."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Tokens produced per target forward pass (1.0 for ordinary decoding)."""
        return self.emitted / self.rounds if self.rounds else 0.0


def verify_greedy(
    draft_tokens: Sequence[int], target_choices: Sequence[int]
) -> tuple[list[int], int]:
    """Keep draft tokens while they match the target's argmax, then take the target's token.

    ``target_choices[i]`` is the target's greedy pick at draft position ``i``; it has
    one more entry than ``draft_tokens`` for the bonus position. Returns the tokens to
    append and how many draft tokens were accepted.
    """
    if len(target_choices) != len(draft_tokens) + 1:
        raise ValueError("need one target choice per draft token plus one")
    for i, token in enumerate(draft_tokens):
        if token != target_choices[i]:
            return [*draft_tokens[:i], target_choices[i]], i
    return [*draft_tokens, target_choices[-1]], len(draft_tokens)


def verify_sampled(
    draft_tokens: Sequence[int],
    draft_probs: Tensor,
    target_probs: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """The speculative sampling accept/reject rule.

    ``draft_probs`` is ``(k, V)``: the distribution each draft token was sampled from.
    ``target_probs`` is ``(k + 1, V)``: the target's distribution at the same positions
    plus one more. Returns the tokens to append and how many draft tokens were accepted.
    """
    k = len(draft_tokens)
    if draft_probs.shape[0] != k or target_probs.shape[0] != k + 1:
        raise ValueError("expected k draft distributions and k + 1 target distributions")

    out: list[int] = []
    for i, token in enumerate(draft_tokens):
        p = float(target_probs[i, token])
        q = float(draft_probs[i, token])
        u = float(torch.rand((), generator=generator))
        # u < p / q, written to avoid dividing by q; q > 0 because the draft sampled token.
        if u * q < p:
            out.append(token)
            continue
        residual = (target_probs[i] - draft_probs[i]).clamp(min=0.0)
        mass = float(residual.sum())
        # Rejection implies p(x) < q(x), so p != q and the residual has positive mass.
        # Guard against round-off anyway by falling back to p itself.
        dist = residual / mass if mass > 0 else target_probs[i]
        out.append(int(torch.multinomial(dist, 1, generator=generator)))
        return out, i
    out.append(int(torch.multinomial(target_probs[k], 1, generator=generator)))
    return out, k


def speculative_generate(
    target: GPT,
    draft: GPT,
    prompt: Sequence[int],
    max_new_tokens: int,
    sampling: SamplingConfig = GREEDY,
    *,
    k: int = 4,
    generator: torch.Generator | None = None,
    stats: SpeculativeStats | None = None,
) -> Iterator[int]:
    """Yield up to ``max_new_tokens`` tokens using ``draft`` to propose and ``target`` to verify.

    The models can differ in depth and width but must share a tokenizer. Pass a
    :class:`SpeculativeStats` to collect the acceptance rate.
    """
    if not prompt:
        raise ValueError("prompt must contain at least one token")
    if k < 1:
        raise ValueError("k must be >= 1")
    if target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("draft and target must share a vocabulary")
    if k + 1 > min(target.config.block_size, draft.config.block_size) // 2:
        raise ValueError("k + 1 must fit in half of the smaller context window")

    target.eval()
    draft.eval()
    stats = stats if stats is not None else SpeculativeStats()
    target_dec = IncrementalDecoder(target)
    draft_dec = IncrementalDecoder(draft)
    seq = list(prompt)
    produced = 0

    while produced < max_new_tokens:
        # A round emits at most n_draft + 1 tokens, so don't draft past the end.
        n_draft = min(k, max_new_tokens - produced - 1)
        with torch.inference_mode():
            new_tokens, n_accepted = _speculative_round(
                target_dec, draft_dec, seq, n_draft, sampling, generator
            )
        seq.extend(new_tokens)
        # Everything up to the last emitted token is now agreed; the last token itself
        # has not been fed to either model yet.
        target_dec.rollback(len(seq) - 1)
        draft_dec.rollback(len(seq) - 1)

        stats.rounds += 1
        stats.proposed += n_draft
        stats.accepted += n_accepted
        stats.emitted += len(new_tokens)
        for token in new_tokens:
            produced += 1
            yield token


def _speculative_round(
    target_dec: IncrementalDecoder,
    draft_dec: IncrementalDecoder,
    seq: list[int],
    n_draft: int,
    sampling: SamplingConfig,
    generator: torch.Generator | None,
) -> tuple[list[int], int]:
    # 1. Draft n_draft tokens autoregressively with the small model.
    drafted: list[int] = []
    draft_rows: list[Tensor] = []
    for _ in range(n_draft):
        context = seq + drafted
        logits = draft_dec.logits(context)[-1].float().cpu()
        if sampling.greedy:
            token = int(process_logits(logits, sampling, context).argmax())
        else:
            q = next_token_probs(logits, sampling, context)
            token = int(torch.multinomial(q, 1, generator=generator))
            draft_rows.append(q)
        drafted.append(token)

    # 2. Score the last agreed token and every draft token in one target pass.
    #    Row i predicts the token that follows seq + drafted[:i].
    target_logits = target_dec.logits(seq + drafted, n_last=n_draft + 1).float().cpu()
    histories = [seq + drafted[:i] for i in range(n_draft + 1)]

    # 3. Accept or reject.
    if sampling.greedy:
        choices = [
            int(process_logits(target_logits[i], sampling, histories[i]).argmax())
            for i in range(n_draft + 1)
        ]
        return verify_greedy(drafted, choices)
    target_probs = torch.stack(
        [next_token_probs(target_logits[i], sampling, histories[i]) for i in range(n_draft + 1)]
    )
    draft_probs = (
        torch.stack(draft_rows) if draft_rows else target_probs.new_zeros((0, target_probs.size(1)))
    )
    return verify_sampled(drafted, draft_probs, target_probs, generator)
