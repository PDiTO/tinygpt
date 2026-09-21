from pathlib import Path

import pytest

from tinygpt.checkpoint import load_checkpoint
from tinygpt.cli import main


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory, fixture_path: Path) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("ckpt")
    target, draft = out / "target.pt", out / "draft.pt"
    common = ["--preset", "tiny", "--data", str(fixture_path), "--device", "cpu"]
    target_args = ["--steps", "30", "--eval-interval", "10", "--out", str(target)]
    # A peak LR below the preset's min_lr must still work: min_lr follows it down.
    draft_args = ["--steps", "10", "--seed", "1", "--lr", "1e-4", "--out", str(draft)]
    assert main(["train", *common, *target_args]) == 0
    assert main(["train", *common, *draft_args]) == 0
    return target, draft


def test_train_writes_a_loadable_checkpoint(checkpoints: tuple[Path, Path]) -> None:
    ckpt = load_checkpoint(checkpoints[0])
    assert ckpt.metadata["step"] > 0
    assert ckpt.model.config.n_layer == 2


def test_sample_streams_text(
    checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sample", str(checkpoints[0]), "--prompt", "First", "-n", "40"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("First")
    assert len(out) == len("First") + 40 + 1


def test_greedy_sample_is_the_same_with_and_without_cache(
    checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["sample", str(checkpoints[0]), "--prompt", "All:", "-n", "30", "--greedy"]
    main(args)
    cached = capsys.readouterr().out
    main([*args, "--no-cache"])
    assert capsys.readouterr().out == cached


def test_speculative_sample_reports_acceptance(
    checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    target, draft = checkpoints
    args = ["sample", str(target), "--prompt", "All:", "-n", "30", "--greedy"]
    main(args)
    plain = capsys.readouterr().out
    assert main([*args, "--draft", str(draft), "-k", "3"]) == 0
    captured = capsys.readouterr()
    assert captured.out == plain
    assert "acceptance" in captured.err


def test_sample_rejects_unknown_characters(
    checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sample", str(checkpoints[0]), "--prompt", "~~~"]) == 2
    assert "not in vocabulary" in capsys.readouterr().err


def test_bench_prints_a_table(
    checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    target, draft = checkpoints
    args = ["bench", "--target", str(target), "--draft", str(draft), "--tokens", "20"]
    assert main([*args, "--repeats", "1", "-k", "2", "3", "--prompt", "All:"]) == 0
    out = capsys.readouterr().out
    for row in ["target, no cache", "target, KV cache", "speculative, k=2", "speculative, k=3"]:
        assert out.count(row) == 2  # greedy and sampled
    assert "| greedy | target, KV cache |" in out


def test_bench_rejects_bad_input(
    checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["bench", "--target", str(checkpoints[0]), "--prompt", "~~~"]) == 2
    assert "not in vocabulary" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["bench", "--target", str(checkpoints[0]), "--tokens", "0"])
