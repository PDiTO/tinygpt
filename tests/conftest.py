from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tinygpt.config import ModelConfig
from tinygpt.model import GPT
from tinygpt.tokenizer import CharTokenizer
from tinygpt.train import TrainConfig, train

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "shakespeare_excerpt.txt"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return FIXTURE_PATH


@pytest.fixture(scope="session")
def fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


@dataclass(frozen=True)
class ModelPair:
    """A small target and an even smaller draft, both briefly trained on the fixture."""

    target: GPT
    draft: GPT
    tokenizer: CharTokenizer


def _train_small(tok: CharTokenizer, text: str, n_layer: int, seed: int) -> GPT:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size, block_size=64, n_layer=n_layer, n_head=2, n_embd=32
    )
    model = GPT(cfg)
    train_cfg = TrainConfig(
        max_steps=200,
        batch_size=16,
        learning_rate=1e-2,
        min_lr=1e-3,
        warmup_steps=10,
        eval_interval=1_000,
        eval_iters=1,
        seed=seed,
    )
    train(model, tok, text, train_cfg, log=lambda _: None)
    return model.eval()


@pytest.fixture(scope="session")
def trained_pair(fixture_text: str) -> ModelPair:
    # Randomly initialised models tend to repeat their input token forever (the tied
    # embedding makes "predict the last token" the path of least resistance), which
    # would make any draft look perfect. A couple of hundred steps fixes that.
    tok = CharTokenizer.from_text(fixture_text)
    return ModelPair(
        target=_train_small(tok, fixture_text, n_layer=2, seed=0),
        draft=_train_small(tok, fixture_text, n_layer=1, seed=1),
        tokenizer=tok,
    )
