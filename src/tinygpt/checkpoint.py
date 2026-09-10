"""Save and load a model together with its tokenizer.

A checkpoint is a single ``torch.save`` file containing only plain Python
containers and tensors, so it loads with ``weights_only=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from tinygpt.config import ModelConfig
from tinygpt.model import GPT
from tinygpt.tokenizer import Tokenizer, tokenizer_from_dict

FORMAT_VERSION = 1


@dataclass
class Checkpoint:
    model: GPT
    tokenizer: Tokenizer
    metadata: dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    model: GPT,
    tokenizer: Tokenizer,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": model.config.to_dict(),
        "model_state": state,
        "tokenizer": tokenizer.to_dict(),
        "metadata": metadata or {},
    }
    # Write then rename so an interrupted save never leaves a truncated checkpoint.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Checkpoint:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format version: {version!r}")
    config = ModelConfig.from_dict(payload["model_config"])
    model = GPT(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    tokenizer = tokenizer_from_dict(payload["tokenizer"])
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"tokenizer vocab ({tokenizer.vocab_size}) does not match model ({config.vocab_size})"
        )
    return Checkpoint(model=model, tokenizer=tokenizer, metadata=dict(payload["metadata"]))
