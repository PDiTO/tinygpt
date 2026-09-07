from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "shakespeare_excerpt.txt"


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return FIXTURE_PATH


@pytest.fixture(scope="session")
def fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")
