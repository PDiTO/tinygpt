"""Dataset download and batching."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import torch
from torch import Tensor

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
SHAKESPEARE_FILENAME = "tinyshakespeare.txt"


def cache_dir() -> Path:
    """Where downloaded data lives: ``$TINYGPT_CACHE_DIR``, else ``$XDG_CACHE_HOME/tinygpt``."""
    if env := os.environ.get("TINYGPT_CACHE_DIR"):
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "tinygpt"


def download_shakespeare(directory: Path | None = None, *, timeout: float = 30.0) -> Path:
    """Fetch tiny Shakespeare (about 1.1 MB) once and verify its checksum."""
    directory = directory or cache_dir()
    path = directory / SHAKESPEARE_FILENAME
    if path.exists() and _sha256(path.read_bytes()) == SHAKESPEARE_SHA256:
        return path

    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SHAKESPEARE_URL, timeout=timeout) as response:
        payload: bytes = response.read()
    digest = _sha256(payload)
    if digest != SHAKESPEARE_SHA256:
        raise RuntimeError(f"checksum mismatch for {SHAKESPEARE_URL}: got {digest}")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path


def load_text(source: str) -> str:
    """``"shakespeare"`` downloads the default corpus; anything else is treated as a file path."""
    path = download_shakespeare() if source == "shakespeare" else Path(source)
    return path.read_text(encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def train_val_split(tokens: Tensor, val_fraction: float) -> tuple[Tensor, Tensor]:
    """Split a 1D token tensor into a leading train part and a trailing validation part."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(len(tokens) * val_fraction))
    return tokens[:-n_val], tokens[-n_val:]


def get_batch(
    data: Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Random contiguous windows. Targets are the inputs shifted left by one."""
    if len(data) <= block_size:
        raise ValueError(f"need more than {block_size} tokens, got {len(data)}")
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x.to(device), y.to(device)
