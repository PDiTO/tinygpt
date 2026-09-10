"""Training loop: AdamW, linear warmup into cosine decay, gradient clipping, periodic eval."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from tinygpt.checkpoint import save_checkpoint
from tinygpt.data import get_batch, train_val_split
from tinygpt.model import GPT
from tinygpt.tokenizer import Tokenizer


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 20
    val_fraction: float = 0.1
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.batch_size <= 0:
            raise ValueError("max_steps and batch_size must be positive")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr must not exceed learning_rate")


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to ``learning_rate``, then cosine decay to ``min_lr`` at ``max_steps``."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def make_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices (linear weights, embedding) only; not on norm gains."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))


def lm_loss(model: GPT, x: Tensor, y: Tensor) -> Tensor:
    logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def estimate_loss(
    model: GPT, data: Tensor, cfg: TrainConfig, generator: torch.Generator, device: torch.device
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, model.config.block_size, cfg.batch_size, generator, device)
        losses.append(lm_loss(model, x, y).item())
    model.train(was_training)
    return sum(losses) / len(losses)


@dataclass
class EvalRecord:
    step: int
    train_loss: float
    val_loss: float


@dataclass
class TrainResult:
    evals: list[EvalRecord] = field(default_factory=list)
    step_losses: list[float] = field(default_factory=list)
    best_val_loss: float = math.inf
    seconds: float = 0.0


def train(
    model: GPT,
    tokenizer: Tokenizer,
    text: str,
    cfg: TrainConfig,
    *,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train ``model`` in place. Saves the best-validation checkpoint if a path is given."""
    device = torch.device(device)
    torch.manual_seed(cfg.seed)
    batch_gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)

    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = train_val_split(tokens, cfg.val_fraction)
    block_size = model.config.block_size

    model.to(device).train()
    optimizer = make_optimizer(model, cfg)
    result = TrainResult()
    log(
        f"params {model.num_params():,} | train tokens {len(train_data):,} | "
        f"val tokens {len(val_data):,} | device {device}"
    )

    def evaluate(step: int) -> None:
        train_loss = estimate_loss(model, train_data, cfg, eval_gen, device)
        val_loss = estimate_loss(model, val_data, cfg, eval_gen, device)
        result.evals.append(EvalRecord(step, train_loss, val_loss))
        note = ""
        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            if checkpoint_path is not None:
                meta = {"step": step, "val_loss": val_loss, "train_config": asdict(cfg)}
                save_checkpoint(checkpoint_path, model, tokenizer, meta)
                note = " (saved)"
        elapsed = time.perf_counter() - start
        log(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:6.1f}s{note}")

    start = time.perf_counter()
    for step in range(cfg.max_steps):
        if step % cfg.eval_interval == 0:
            evaluate(step)

        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, block_size, cfg.batch_size, batch_gen, device)
        loss = lm_loss(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        result.step_losses.append(loss.item())

    evaluate(cfg.max_steps)
    result.seconds = time.perf_counter() - start
    model.eval()
    return result
