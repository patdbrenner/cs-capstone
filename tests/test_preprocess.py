from __future__ import annotations

from src.preprocess import clean_text


def test_clean_text_lowercases() -> None:
    assert clean_text("HELLO World") == clean_text("hello world")

def test_clean_text_removes_punctuation() -> None:
    result = clean_text("Can't connect to VPN!")
    assert "'" not in result
    assert "!" not in result

def test_clean_text_removes_stop_words() -> None:
    result = clean_text("this is a test of the system")
    for stop_word in ("is", "a", "of", "the"):
        assert stop_word not in result.split()

def test_clean_text_applies_stemming() -> None:
    result = clean_text("running runner runs")
    tokens = result.split()
    assert all(token.startswith("run") for token in tokens)

def test_clean_text_handles_non_string_input() -> None:
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_clean_text_handles_empty_string() -> None:
    assert clean_text("") == ""