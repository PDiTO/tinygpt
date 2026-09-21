import pytest
import torch

from tinygpt.config import ModelConfig
from tinygpt.generate import IncrementalDecoder, generate
from tinygpt.kv_cache import KVCache
from tinygpt.model import GPT
from tinygpt.sampling import GREEDY, SamplingConfig

VOCAB = 23


def make_model(seed: int = 0, block_size: int = 48) -> GPT:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=block_size, n_layer=3, n_head=4, n_embd=32)
    model = GPT(cfg).eval()
    # Spread the logits out so greedy comparisons aren't decided by near-ties.
    with torch.no_grad():
        model.tok_emb.weight.normal_(0.0, 0.3)
    return model


@pytest.mark.parametrize("chunks", [[1] * 20, [5, 1, 1, 1, 4, 8], [20], [7, 13]])
def test_incremental_logits_match_full_forward(chunks: list[int]) -> None:
    model = make_model()
    idx = torch.randint(0, VOCAB, (2, sum(chunks)), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = model(idx)
        cache = model.new_cache(batch_size=2)
        pieces, start = [], 0
        for size in chunks:
            pieces.append(model(idx[:, start : start + size], cache=cache))
            start += size
        incremental = torch.cat(pieces, dim=1)
    assert cache.pos == idx.size(1)
    torch.testing.assert_close(incremental, full, rtol=1e-5, atol=1e-5)


def test_crop_then_continue_matches_full_forward_of_new_sequence() -> None:
    model = make_model()
    gen = torch.Generator().manual_seed(2)
    prefix = torch.randint(0, VOCAB, (1, 10), generator=gen)
    old_tail = torch.randint(0, VOCAB, (1, 6), generator=gen)
    new_tail = torch.randint(0, VOCAB, (1, 4), generator=gen)
    with torch.no_grad():
        cache = model.new_cache()
        model(torch.cat([prefix, old_tail], dim=1), cache=cache)
        cache.crop(10)
        out = model(new_tail, cache=cache)
        expected = model(torch.cat([prefix, new_tail], dim=1))[:, 10:]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("seed", "prompt_len"), [(0, 1), (1, 5), (2, 17), (3, 30)])
def test_cached_greedy_equals_uncached_greedy(seed: int, prompt_len: int) -> None:
    model = make_model(seed)
    prompt = torch.randint(0, VOCAB, (prompt_len,), generator=torch.Generator().manual_seed(seed))
    n_new = model.config.block_size - prompt_len
    cached = list(generate(model, prompt.tolist(), n_new, GREEDY, use_cache=True))
    uncached = list(generate(model, prompt.tolist(), n_new, GREEDY, use_cache=False))
    assert len(cached) == n_new
    assert cached == uncached


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cached_sampling_equals_uncached_sampling(seed: int) -> None:
    model = make_model(seed)
    cfg = SamplingConfig(temperature=0.9, top_k=10, top_p=0.95, repetition_penalty=1.1)

    def run(use_cache: bool) -> list[int]:
        gen = torch.Generator().manual_seed(seed)
        return list(generate(model, [1, 2, 3], 40, cfg, use_cache=use_cache, generator=gen))

    assert run(True) == run(False)


def test_decoder_feeds_one_token_per_step() -> None:
    model = make_model()
    decoder = IncrementalDecoder(model)
    assert decoder.cache is not None
    seq = [1, 2, 3, 4]
    with torch.no_grad():
        decoder.logits(seq)
        assert decoder.cache.pos == 4
        seq.append(5)
        decoder.logits(seq)
        assert decoder.cache.pos == 5


def test_decoder_refuses_to_reuse_stale_positions() -> None:
    model = make_model()
    decoder = IncrementalDecoder(model)
    with torch.no_grad():
        decoder.logits([1, 2, 3, 4, 5])
        with pytest.raises(RuntimeError, match="rollback"):
            decoder.logits([1, 2, 3, 4, 5], n_last=2)
        decoder.rollback(3)
        out = decoder.logits([1, 2, 3, 4, 5], n_last=2)
        expected = model(torch.tensor([[1, 2, 3, 4, 5]]))[0, -2:]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_cached_generation_slides_past_the_context_window() -> None:
    model = make_model(block_size=16)
    tokens = list(generate(model, [1, 2, 3], 50, GREEDY, use_cache=True))
    assert len(tokens) == 50
    assert all(0 <= t < VOCAB for t in tokens)


def test_cache_overflow_raises() -> None:
    model = make_model(block_size=8)
    cache = model.new_cache()
    with torch.no_grad():
        model(torch.zeros(1, 6, dtype=torch.long), cache=cache)
        with pytest.raises(ValueError, match="exceeds block_size"):
            model(torch.zeros(1, 3, dtype=torch.long), cache=cache)


def test_kv_cache_bookkeeping() -> None:
    cache = KVCache(n_layer=2, batch_size=1, n_head=2, max_len=5, head_dim=4)
    k = torch.ones(1, 2, 3, 4)
    keys, values = cache.update(0, k, k * 2)
    assert keys.shape == (1, 2, 3, 4)
    torch.testing.assert_close(values, k * 2)
    cache.advance(3)
    assert len(cache) == 3
    cache.crop(1)
    assert len(cache) == 1
    with pytest.raises(ValueError, match="cannot crop"):
        cache.crop(2)
    with pytest.raises(ValueError, match="overflow"):
        cache.update(0, torch.ones(1, 2, 5, 4), torch.ones(1, 2, 5, 4))
    cache.reset()
    assert len(cache) == 0


def test_rollback_past_the_window_start_recovers() -> None:
    model = make_model(block_size=16)
    decoder = IncrementalDecoder(model)
    seq = list(range(20))
    with torch.no_grad():
        decoder.logits(seq)  # slides: the cache now starts part-way through seq
        assert decoder.offset > 5
        decoder.rollback(5)
        out = decoder.logits(seq[:6])
        expected = model(torch.tensor([seq[:6]]))[0, -1:]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
