"""Named model + training configurations used by the CLI.

``target`` and ``draft`` share a block size so the draft can be used for
speculative decoding against the target. Both train in a few minutes on an
Apple Silicon laptop (MPS); ``tiny`` exists for smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygpt.train import TrainConfig


@dataclass(frozen=True)
class Preset:
    model: dict[str, Any]
    train: TrainConfig


PRESETS: dict[str, Preset] = {
    "target": Preset(
        model={"block_size": 256, "n_layer": 6, "n_head": 6, "n_embd": 192, "dropout": 0.1},
        train=TrainConfig(
            max_steps=3000,
            batch_size=16,
            learning_rate=1.5e-3,
            min_lr=1.5e-4,
            warmup_steps=100,
            eval_interval=250,
        ),
    ),
    "draft": Preset(
        model={"block_size": 256, "n_layer": 1, "n_head": 4, "n_embd": 128},
        train=TrainConfig(
            max_steps=1500,
            batch_size=32,
            learning_rate=2e-3,
            min_lr=2e-4,
            warmup_steps=100,
            eval_interval=250,
        ),
    ),
    "tiny": Preset(
        model={"block_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 32},
        train=TrainConfig(
            max_steps=200,
            batch_size=16,
            learning_rate=3e-3,
            min_lr=3e-4,
            warmup_steps=20,
            eval_interval=50,
            eval_iters=5,
        ),
    ),
}
