import hashlib
import io
from pathlib import Path

import pytest

from tinygpt import data


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_download_verifies_checksum_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"To be, or not to be\n"
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> FakeResponse:
        calls.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(data, "SHAKESPEARE_SHA256", hashlib.sha256(payload).hexdigest())

    path = data.download_shakespeare(tmp_path)
    assert path.read_bytes() == payload
    data.download_shakespeare(tmp_path)
    assert len(calls) == 1  # second call is served from the cache


def test_download_rejects_a_corrupted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url, timeout: FakeResponse(b"not shakespeare")
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.download_shakespeare(tmp_path)
    assert not (tmp_path / data.SHAKESPEARE_FILENAME).exists()


def test_cache_dir_respects_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYGPT_CACHE_DIR", str(tmp_path / "custom"))
    assert data.cache_dir() == tmp_path / "custom"
    monkeypatch.delenv("TINYGPT_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert data.cache_dir() == tmp_path / "xdg" / "tinygpt"


def test_load_text_reads_a_path(fixture_path: Path) -> None:
    assert data.load_text(str(fixture_path)).startswith("First Citizen:")
