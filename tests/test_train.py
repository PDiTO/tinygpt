import itertools
import math
from pathlib import Path

import pytest
import torch

from tinygpt.checkpoint import load_checkpoint, save_checkpoint
from tinygpt.config import ModelConfig
from tinygpt.data import get_batch, train_val_split
from tinygpt.model import GPT
from tinygpt.tokenizer import ByteTokenizer, CharTokenizer
from tinygpt.train import TrainConfig, lr_at, make_optimizer, train


def test_lr_schedule_warmup_then_cosine() -> None:
    cfg = TrainConfig(max_steps=100, learning_rate=1.0, min_lr=0.1, warmup_steps=10)
    lrs = [lr_at(s, cfg) for s in range(cfg.max_steps + 1)]
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[9] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(1.0)
    # Halfway through the decay the cosine sits at the midpoint.
    assert lrs[55] == pytest.approx(0.55)
    assert lrs[100] == pytest.approx(0.1)
    decay = lrs[10:]
    assert all(a >= b for a, b in itertools.pairwise(decay))


def test_optimizer_skips_weight_decay_on_norm_gains() -> None:
    model = GPT(ModelConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16))
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() == 1 for p in no_decay["params"])
    n_grouped = len(decay["params"]) + len(no_decay["params"])
    assert n_grouped == len(list(model.parameters()))


def test_get_batch_targets_are_shifted_inputs() -> None:
    data = torch.arange(100)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    torch.testing.assert_close(y, x + 1)


def test_train_val_split_is_contiguous() -> None:
    train_part, val_part = train_val_split(torch.arange(100), 0.1)
    assert train_part.tolist() == list(range(90))
    assert val_part.tolist() == list(range(90, 100))


def test_short_training_run_reduces_loss(fixture_text: str, tmp_path: Path) -> None:
    tok = CharTokenizer.from_text(fixture_text)
    model = GPT(
        ModelConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32)
    )
    cfg = TrainConfig(
        max_steps=150,
        batch_size=16,
        learning_rate=3e-3,
        min_lr=3e-4,
        warmup_steps=10,
        eval_interval=50,
        eval_iters=5,
        seed=0,
    )
    ckpt = tmp_path / "model.pt"
    result = train(model, tok, fixture_text, cfg, checkpoint_path=ckpt, log=lambda _: None)

    first, last = result.evals[0], result.evals[-1]
    # Untrained loss is close to uniform over the vocab.
    assert first.train_loss == pytest.approx(math.log(tok.vocab_size), abs=0.3)
    assert last.train_loss < first.train_loss - 1.0
    assert last.val_loss < first.val_loss - 0.8
    assert ckpt.exists()
    assert load_checkpoint(ckpt).metadata["val_loss"] == pytest.approx(result.best_val_loss)


@pytest.mark.parametrize("tokenizer", [CharTokenizer.from_text("abcdefgh"), ByteTokenizer()])
def test_checkpoint_round_trip(tmp_path: Path, tokenizer: CharTokenizer | ByteTokenizer) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=16
    )
    model = GPT(cfg).eval()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer, {"step": 7})

    loaded = load_checkpoint(path)
    assert loaded.model.config == cfg
    assert loaded.metadata == {"step": 7}
    assert loaded.tokenizer.to_dict() == tokenizer.to_dict()
    # Tying survives the round trip.
    assert loaded.model.lm_head.weight is loaded.model.tok_emb.weight

    idx = torch.randint(0, tokenizer.vocab_size, (2, 10))
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(idx), model(idx), rtol=0, atol=0)


def test_checkpoint_rejects_mismatched_tokenizer(tmp_path: Path) -> None:
    model = GPT(ModelConfig(vocab_size=5, block_size=8, n_layer=1, n_head=2, n_embd=8))
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model, CharTokenizer.from_text("abc"))
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(path)
