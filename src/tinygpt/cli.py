"""Command line interface: ``tinygpt train``, ``tinygpt sample`` and ``tinygpt bench``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import torch

from tinygpt.bench import format_table, run_benchmarks
from tinygpt.checkpoint import load_checkpoint
from tinygpt.config import ModelConfig
from tinygpt.data import load_text
from tinygpt.generate import generate
from tinygpt.model import GPT
from tinygpt.presets import PRESETS
from tinygpt.sampling import SamplingConfig
from tinygpt.speculative import SpeculativeStats, speculative_generate
from tinygpt.tokenizer import ByteTokenizer, CharTokenizer, Tokenizer
from tinygpt.train import train
from tinygpt.utils import resolve_device


def cmd_train(args: argparse.Namespace) -> int:
    preset = PRESETS[args.preset]
    text = load_text(args.data)
    tokenizer: Tokenizer = (
        ByteTokenizer() if args.tokenizer == "byte" else CharTokenizer.from_text(text)
    )

    overrides = {
        "max_steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "eval_interval": args.eval_interval,
        "seed": args.seed,
    }
    train_cfg = replace(preset.train, **{k: v for k, v in overrides.items() if v is not None})
    if args.lr is not None:
        train_cfg = replace(train_cfg, min_lr=args.lr / 10)
    model_cfg = ModelConfig(vocab_size=tokenizer.vocab_size, **preset.model)

    torch.manual_seed(train_cfg.seed)
    model = GPT(model_cfg)
    out = Path(args.out) if args.out else Path("checkpoints") / f"{args.preset}.pt"
    result = train(
        model,
        tokenizer,
        text,
        train_cfg,
        device=resolve_device(args.device),
        checkpoint_path=out,
        log=lambda line: print(line, flush=True),
    )
    print(
        f"done in {result.seconds:.1f}s, best val loss {result.best_val_loss:.4f}, saved to {out}"
    )
    return 0


def sampling_from_args(args: argparse.Namespace) -> SamplingConfig:
    if args.greedy:
        return SamplingConfig(temperature=0.0, repetition_penalty=args.repetition_penalty)
    return SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def cmd_sample(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    tokenizer = ckpt.tokenizer
    try:
        prompt = tokenizer.encode(args.prompt)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not prompt:
        print("error: prompt must not be empty", file=sys.stderr)
        return 2

    sampling = sampling_from_args(args)
    generator = torch.Generator().manual_seed(args.seed)
    stats: SpeculativeStats | None = None
    if args.draft:
        draft = load_checkpoint(args.draft, device)
        if draft.tokenizer.to_dict() != tokenizer.to_dict():
            print("error: draft and target were trained with different tokenizers", file=sys.stderr)
            return 2
        stats = SpeculativeStats()
        tokens = speculative_generate(
            ckpt.model,
            draft.model,
            prompt,
            args.max_new_tokens,
            sampling,
            k=args.spec_k,
            generator=generator,
            stats=stats,
        )
    else:
        tokens = generate(
            ckpt.model,
            prompt,
            args.max_new_tokens,
            sampling,
            use_cache=not args.no_cache,
            generator=generator,
        )

    sys.stdout.write(args.prompt)
    for piece in tokenizer.decode_stream(tokens):
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")
    if stats is not None:
        print(
            f"[speculative k={args.spec_k}] acceptance {stats.acceptance_rate:.1%}, "
            f"{stats.tokens_per_round:.2f} tokens per target pass",
            file=sys.stderr,
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    target = load_checkpoint(args.target, device)
    draft = load_checkpoint(args.draft, device) if args.draft else None
    if draft is not None and draft.tokenizer.to_dict() != target.tokenizer.to_dict():
        print("error: draft and target were trained with different tokenizers", file=sys.stderr)
        return 2
    prompt = target.tokenizer.encode(args.prompt)
    context = len(prompt) + args.tokens
    if context > target.model.config.block_size:
        print(
            f"note: {context} tokens exceeds the context window "
            f"({target.model.config.block_size}); cached decoding will slide the window",
            file=sys.stderr,
        )

    print(
        f"device {device}, torch threads {torch.get_num_threads()}, "
        f"target {target.model.num_params():,} params"
        + (f", draft {draft.model.num_params():,} params" if draft else "")
        + f", {args.tokens} new tokens, median of {args.repeats}",
        flush=True,
    )
    results = run_benchmarks(
        target.model,
        draft.model if draft else None,
        prompt,
        args.tokens,
        repeats=args.repeats,
        spec_ks=args.spec_k,
        temperature=args.temperature,
        device=device,
        progress=lambda msg: print(f"  running {msg}", file=sys.stderr, flush=True),
    )
    print(format_table(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinygpt", description="A small GPT for studying KV caching and speculative decoding."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train a model and save the best checkpoint")
    p.add_argument("--preset", choices=sorted(PRESETS), default="target")
    p.add_argument(
        "--data", default="shakespeare", help='"shakespeare" (downloaded once) or a text file path'
    )
    p.add_argument("--tokenizer", choices=["char", "byte"], default="char")
    p.add_argument("--out", help="checkpoint path (default: checkpoints/<preset>.pt)")
    p.add_argument("--steps", type=int, help="override the preset's step count")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, help="peak learning rate; min lr becomes lr / 10")
    p.add_argument("--eval-interval", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sample", help="generate text from a checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default="\n")
    p.add_argument("-n", "--max-new-tokens", type=int, default=500)
    add_sampling_args(p)
    p.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    p.add_argument("--draft", help="draft checkpoint: enables speculative decoding")
    p.add_argument("-k", "--spec-k", type=int, default=4, help="draft tokens per round")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("bench", help="measure decoding speed with and without the tricks")
    p.add_argument("--target", required=True, help="target checkpoint")
    p.add_argument("--draft", help="draft checkpoint; adds speculative decoding rows")
    p.add_argument("--tokens", type=int, default=200, help="new tokens per run")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("-k", "--spec-k", type=int, nargs="+", default=[4])
    p.add_argument("--temperature", type=float, default=0.8, help="for the sampled rows")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, help="torch intra-op threads (default: torch's choice)")
    p.set_defaults(func=cmd_bench)

    return parser


def add_sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="argmax decoding (ignores temperature)")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
