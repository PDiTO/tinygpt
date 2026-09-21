# tinygpt

A small decoder-only transformer written from scratch in PyTorch. I built it to poke at the
inference side of autoregressive models. I wanted to see what a decoding step actually does,
what the KV cache saves, and whether speculative decoding really gives you the same output
as the big model on its own.

No `nn.Transformer`, no `scaled_dot_product_attention`, no Hugging Face. Attention, RoPE,
the cache and the speculative accept/reject rule are all written out by hand so every step
is visible and testable.

## Why I built this

I'd read plenty about KV caching and speculative decoding, and I understood them at the
whiteboard level. What I didn't have was a feel for the details. How does the causal mask
change when the queries start at position 200 instead of 0? What exactly do you roll back
when the target rejects a draft token? And the claim that speculative sampling is lossless,
that the output is distributed exactly as if the target had sampled alone, is the kind of
thing I wanted to check with a test rather than take on faith.

So I kept the model tiny and put the effort into the decoding loop and the tests around it.
It's character-level and trains in about four minutes on a laptop.

## What's in here

```
src/tinygpt/
  model.py        RMSNorm, RoPE, causal self-attention, SwiGLU, the GPT module
  kv_cache.py     preallocated per-layer key/value cache with crop/rollback
  generate.py     IncrementalDecoder and the streaming generate() loop
  sampling.py     repetition penalty, temperature, top-k, top-p, greedy
  speculative.py  draft/verify loop and the accept/reject rules
  train.py        AdamW, warmup + cosine LR, grad clipping, eval, checkpointing
  tokenizer.py    character and byte tokenizers with save/load
  data.py         tiny Shakespeare download (cached, checksummed) and batching
  bench.py        throughput measurements behind `tinygpt bench`
  cli.py          `tinygpt train | sample | bench`
tests/            pytest suite, runs offline on CPU in about 10 seconds
```

## Quickstart

You need [uv](https://docs.astral.sh/uv/). The project pins Python 3.14 in
`.python-version` and requires 3.12 or newer. PyTorch is CPU-only. On Linux it comes from
the PyTorch CPU wheel index, and on macOS the regular PyPI wheel is already CPU/MPS only.

```sh
uv sync

# Train the 6-layer target (~4 min on MPS) and the 1-layer draft (~30 s).
# Tiny Shakespeare (1.1 MB) downloads to ~/.cache/tinygpt on first use.
uv run tinygpt train --preset target
uv run tinygpt train --preset draft

# Sample
uv run tinygpt sample checkpoints/target.pt --prompt "ROMEO:" -n 400 --temperature 0.8
uv run tinygpt sample checkpoints/target.pt --prompt "ROMEO:" --top-p 0.9 --repetition-penalty 1.1

# Same thing through speculative decoding
uv run tinygpt sample checkpoints/target.pt --draft checkpoints/draft.pt -k 4 --prompt "ROMEO:"

# Throughput: no cache vs cache vs speculative, greedy and sampled
uv run tinygpt bench --target checkpoints/target.pt --draft checkpoints/draft.pt -k 2 4 8 --repeats 5
```

`train` defaults to `--device auto`, which picks CUDA, then MPS, then CPU. `sample` and
`bench` default to CPU because that turned out to be faster for batch-size-1 decoding here.
More on that below.

CI runs these four checks, and they work the same way locally:

```sh
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## The model

| | target | draft |
|---|---|---|
| layers | 6 | 1 |
| heads | 6 | 4 |
| width | 192 | 128 |
| context | 256 | 256 |
| parameters | 2.67M | 0.21M |
| dropout | 0.1 | 0 |

Both use a character vocabulary of 65 symbols built from the corpus, so they share a
tokenizer, which is the one hard requirement for pairing a draft with a target.

## A training run

This is the actual log from `tinygpt train --preset target` on an Apple Silicon laptop
using MPS:

```
params 2,669,184 | train tokens 1,003,855 | val tokens 111,539 | device mps
step     0 | train 4.2413 | val 4.2429 |    1.8s (saved)
step   500 | train 1.5123 | val 1.7401 |   40.2s (saved)
step  1000 | train 1.3698 | val 1.5779 |   80.4s (saved)
step  1500 | train 1.2742 | val 1.5000 |  121.0s (saved)
step  2000 | train 1.2259 | val 1.4709 |  160.0s (saved)
step  2500 | train 1.1665 | val 1.4740 |  198.9s
step  2750 | train 1.1531 | val 1.4550 |  218.2s (saved)
step  3000 | train 1.1454 | val 1.4587 |  237.4s
done in 237.4s, best val loss 1.4550, saved to checkpoints/target.pt
```

The run logs every 250 steps and I've trimmed some lines. The draft reached a validation
loss of 1.693 in 27 seconds.

Sample from the target, `--temperature 0.8 --seed 1`:

```
ROMEO:
Call'd you be much leave at all, and shall win me.

LADY CAPULET:
To raim on, and what he shall not grant to tell
But to the steels as for the devil.
He is a deadly same but sweet Romeo!

FLORIZEL:
Why dost thou dost follow away of Dion,
To the pleasure of this sleep-shares him.

KING HENRY VI:
Did I no more made John Hastings,
That shall grieve my king and prophecies,
Cannot be my heart so lies
```

Not Shakespeare, but clearly trying to be. The draft on its own with the same settings is
worse. It opens with "Caw; thy son me us lord." That's fine for a draft, which only needs
to be cheap and often right about the next character.

## Results

Measured with `tinygpt bench` on an Apple Silicon laptop, CPU, PyTorch 2.14, 8 threads.
200 new tokens after the prompt `ROMEO:`, batch size 1, median of 5 runs after a warm-up.
The "vs KV cache" column is relative to the target with the cache, in the same mode.

| mode | method | tokens/s | vs KV cache | acceptance | tokens/target pass |
|---|---|---:|---:|---:|---:|
| greedy | target, no cache | 317 | 0.29x |  |  |
| greedy | target, KV cache | 1,076 | 1.00x |  |  |
| greedy | draft, KV cache | 5,782 | 5.37x |  |  |
| greedy | speculative, k=2 | 1,327 | 1.23x | 59.0% | 2.17 |
| greedy | speculative, k=4 | 1,394 | 1.30x | 44.3% | 2.74 |
| greedy | speculative, k=8 | 1,231 | 1.14x | 28.1% | 3.17 |
| sampled T=0.8 | target, no cache | 310 | 0.28x |  |  |
| sampled T=0.8 | target, KV cache | 1,094 | 1.00x |  |  |
| sampled T=0.8 | draft, KV cache | 5,647 | 5.16x |  |  |
| sampled T=0.8 | speculative, k=2 | 1,351 | 1.23x | 66.7% | 2.32 |
| sampled T=0.8 | speculative, k=4 | 1,715 | 1.57x | 57.0% | 3.25 |
| sampled T=0.8 | speculative, k=8 | 1,411 | 1.29x | 34.4% | 3.72 |

I ran the benchmark twice. Individual throughput numbers moved by a few percent between
runs. The acceptance rates are deterministic for a given seed and came out identical.

What I took away from it:

- **The KV cache is worth about 3.4x here.** That's with a context of only about 200
  tokens. Without it, step *t* re-runs attention and the MLP over all *t* tokens, so the
  total work grows quadratically with output length. With it, each step processes one token
  and only attention still looks at the whole history.
- **Speculative decoding helps, but less than I expected from the draft's speed.** The draft
  has 13x fewer parameters yet is only about 5.4x faster, because at batch size 1 a forward pass
  on a model this small is mostly fixed per-call overhead rather than arithmetic. I timed
  the forward passes on their own at context position 100: one token through the target
  takes 0.87 ms, five tokens take 1.25 ms, one token through the draft takes 0.15 ms. A
  greedy round with k=4 is then four draft steps plus one five-token verify, about 1.87 ms,
  and yields 2.74 tokens on average. That's 0.68 ms per token against 0.87 ms, or 1.28x, close
  to the 1.30x in the table.
- **Bigger k isn't better.** Acceptance per drafted token falls as k grows, since one miss
  throws away everything after it, while the draft cost keeps rising linearly. k=4 came out
  best in both modes.
- **Sampling accepted more than greedy did.** At T=0.8 the target's distribution is
  flatter, so `min(1, p/q)` is less often tiny. With greedy decoding a draft token is either
  exactly the target's argmax or it's rejected outright.
- **MPS is the wrong tool for this.** On the same machine with `--device mps` every row
  landed between 110 and 190 tokens/s. The cache made almost no difference, 171 against
  160 tokens/s for greedy, and speculative decoding was slower than plain decoding for
  every k. My read is that each generated token needs a GPU-to-host sync to pick the next
  token, and at this model size that round trip dominates. Training is different. In a
  quick timing of training steps MPS was a bit over 2x faster than CPU, so I trained there.

The acceptance numbers are identical between the CPU and MPS runs, which is a nice sanity
check that the decoding logic doesn't depend on the device.

## How speculative sampling stays exact

Let `p` be the target's next-token distribution and `q` the draft's. Both come out of the
same function, after repetition penalty, temperature, top-k and top-p. The draft samples
`x ~ q`. The target keeps it with probability

```
min(1, p(x) / q(x))
```

and on rejection samples a replacement from the residual

```
r(x) = max(0, p(x) - q(x)) / sum_y max(0, p(y) - q(y))
```

To see why that gives exactly `p`, write out the probability of ending up with token `x`.

```
P(x) = q(x) * min(1, p(x)/q(x))  +  P(reject) * r(x)
     = min(q(x), p(x))           +  (1 - sum_y min(p(y), q(y))) * r(x)
```

and `1 - sum_y min(p, q)` equals `sum_y max(0, p - q)`, the normaliser of `r`. So the
second term is just `max(0, p(x) - q(x))`, and `min(q, p) + max(0, p - q) = p`.

Within a round, the draft proposes `k` tokens, the target scores all of them plus one more
position in a single forward pass, and you walk the proposals left to right applying the
rule. The first rejection ends the round with the residual sample. If all `k` survive you
get a bonus token from the target's last position, so every round emits between 1 and
`k + 1` tokens for one target pass. Both KV caches then roll back to the accepted prefix.

Greedy mode is the same rule with one-hot distributions, which collapses to "keep draft
tokens while they equal the target's argmax, then take the target's argmax".

## Tests

The suite is the part I care most about. It's 135 tests, all offline, about 10 seconds on
CPU. The ones that do the real work:

- **Causal masking.** Perturbing tokens at positions `>= t` leaves logits at positions
  `< t` bit-for-bit unchanged, and the gradient of logit `t` with respect to later input
  embeddings is exactly zero.
- **RoPE.** `<R(m)q, R(n)k> == <R(m+s)q, R(n+s)k>` for arbitrary shifts, rotation preserves
  norms, and position 0 is the identity.
- **KV cache parity.** Feeding a sequence in arbitrary chunks through the cache matches a
  single full forward pass within 1e-5, cropping and re-feeding matches a fresh pass, and
  with the same seed, cached generation equals uncached generation token for token in both
  greedy and sampled mode.
- **Speculative greedy is exact.** For an identical draft, a trained draft and a random
  draft, and k in {1, 2, 4, 7}, speculative greedy output equals target-only greedy output
  exactly.
- **Speculative sampling preserves the target distribution.** The toy target and draft are
  Markov chains over three tokens. I draw 20,000 speculative samples of three tokens each
  and compare them to the target chain's joint distribution with a chi-square test, plus a
  total variation bound of 0.02. The same check also runs against a deliberately wrong rule
  that resamples from `p` instead of the residual after a rejection, and there it has to
  fail. That second test shows the first one can actually catch a mistake. There's also an
  end-to-end version with real trained models, checking the first sampled token follows
  the target and not the draft.
- **Sampling filters.** Hand-built logits check top-k with ties at the cut-off, top-p at
  several thresholds including the edge where the top token must always survive, the sign
  handling in the repetition penalty, temperature scaling, and temperature near 0 matching
  greedy.
- Tokenizer round trips, including multi-byte UTF-8 streaming for the byte tokenizer,
  checkpoint save/load with tied weights, the LR schedule, and a short training run on a
  5.6 KB fixture that has to cut the loss by more than 1 nat.

`ruff check`, `ruff format --check` and `mypy --strict` are clean on both `src` and `tests`.

## Design decisions

- **RoPE instead of learned positions.** Keys are rotated by their absolute position before
  they go into the cache, so cached keys never need touching again and the only thing that
  knows about the offset is the mask slice `causal_mask[start:start+T, :start+T]`.
- **Preallocated cache.** One tensor per K and V shaped `(layers, batch, heads, max_len,
  head_dim)`. Appending is a slice write, and rollback after a rejected draft is moving an
  integer. No concatenation, no reallocation.
- **One `IncrementalDecoder` for both paths.** It tracks which prefix the cache has seen and
  feeds the model only the rest. Speculative decoding uses two of them, and a request for
  logits at positions the cache already covers raises instead of silently returning stale
  results, so a missed rollback shows up as an error rather than as slightly wrong text.
- **One function for the sampling distribution.** `next_token_probs` produces `p` and `q`
  for speculative decoding with the same processing as ordinary sampling. If they drifted
  apart the output would no longer match the target's distribution, and nothing would
  crash.
- **Hand-written attention.** Slower than `scaled_dot_product_attention`, but the point was
  to see the mask arithmetic. Softmax runs in float32.
- **Tied embeddings, SwiGLU, RMSNorm, no biases.** The usual modern choices. Tying barely
  matters with a 65-symbol vocabulary, but it's the standard setup and costs nothing.
- **Weight decay only on matrices**, not on norm gains. Checkpoints contain only tensors and
  plain containers so they load with `torch.load(weights_only=True)`.

## Limitations

- Batch size 1 only for generation. Speculative decoding with batches needs per-row
  acceptance lengths and ragged caches, which is a different project.
- When the sequence outgrows the 256-token context, the cached path drops the oldest half of
  the window and re-fills the cache, while the uncached path slides one token at a time.
  They only agree token for token while everything fits in the window, and the parity tests
  stay inside it.
- The speed numbers are for tiny models where per-call overhead dominates. The speculative
  decoding papers target models big enough to be memory-bandwidth bound, where I'd expect
  the draft to be relatively cheaper and the speed-up larger. I haven't measured that here.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
- Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021.
- Shazeer. *GLU Variants Improve Transformer.* 2020.
- Zhang, Sennrich. *Root Mean Square Layer Normalization.* 2019.
- Holtzman et al. *The Curious Case of Neural Text Degeneration.* 2019. Nucleus sampling.
- Keskar et al. *CTRL: A Conditional Transformer Language Model for Controllable Generation.* 2019. Repetition penalty.
- The tiny Shakespeare corpus from Andrej Karpathy's char-rnn.

## License

MIT. See [LICENSE](LICENSE).
