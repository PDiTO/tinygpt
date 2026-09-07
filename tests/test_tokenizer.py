from pathlib import Path

import pytest

from tinygpt.tokenizer import ByteTokenizer, CharTokenizer, load_tokenizer, tokenizer_from_dict


def test_char_round_trip(fixture_text: str) -> None:
    tok = CharTokenizer.from_text(fixture_text)
    ids = tok.encode(fixture_text)
    assert len(ids) == len(fixture_text)
    assert tok.decode(ids) == fixture_text
    assert max(ids) < tok.vocab_size


def test_char_vocab_is_sorted_and_deterministic() -> None:
    tok = CharTokenizer.from_text("banana")
    assert tok.chars == ["a", "b", "n"]
    assert tok.encode("nab") == [2, 0, 1]


def test_char_unknown_character_raises() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in vocabulary"):
        tok.encode("abz")


def test_char_decode_out_of_range_raises() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([0, 3])


@pytest.mark.parametrize(
    ("chars", "message"),
    [([], "empty"), (["a", "a"], "duplicate"), (["ab"], "single character")],
)
def test_char_rejects_bad_vocab(chars: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CharTokenizer(chars)


def test_byte_round_trip_with_multibyte_text() -> None:
    tok = ByteTokenizer()
    text = "naïve café, 東京 🙂"
    ids = tok.encode(text)
    assert len(ids) > len(text)
    assert all(0 <= i < 256 for i in ids)
    assert tok.decode(ids) == text


def test_byte_stream_decoding_waits_for_complete_characters() -> None:
    tok = ByteTokenizer()
    text = "a東b🙂"
    pieces = list(tok.decode_stream(tok.encode(text)))
    assert "".join(pieces) == text
    # Every piece must be whole characters, never a replacement for a partial sequence.
    assert pieces == ["a", "東", "b", "🙂"]


def test_char_stream_decoding() -> None:
    tok = CharTokenizer.from_text("hello")
    assert list(tok.decode_stream(tok.encode("hello"))) == list("hello")


@pytest.mark.parametrize("tok", [CharTokenizer.from_text("to be or not"), ByteTokenizer()])
def test_save_load_round_trip(tmp_path: Path, tok: CharTokenizer | ByteTokenizer) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = load_tokenizer(path)
    assert type(loaded) is type(tok)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("to be") == tok.encode("to be")
    assert tokenizer_from_dict(tok.to_dict()).to_dict() == tok.to_dict()


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer"):
        tokenizer_from_dict({"kind": "bpe"})
