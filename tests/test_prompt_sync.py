"""Tests for shared prompt sync helpers."""

from app.config.prompt_sync import parse_clean_corrupt_pairs_json


def test_parse_pairs_ok():
    raw = '[["a b", "c d"], ["x", "y"]]'
    pairs, err = parse_clean_corrupt_pairs_json(raw)
    assert err is None
    assert pairs == [["a b", "c d"], ["x", "y"]]


def test_parse_pairs_empty_string():
    pairs, err = parse_clean_corrupt_pairs_json("   ")
    assert pairs is None
    assert err and "empty" in err.lower()


def test_parse_pairs_bad_json():
    pairs, err = parse_clean_corrupt_pairs_json("[")
    assert pairs is None
    assert err and "Invalid JSON" in err


def test_parse_pairs_wrong_shape():
    pairs, err = parse_clean_corrupt_pairs_json('[["only-one-element"]]')
    assert pairs is None
    assert "Entry 0" in err


def test_parse_pairs_inner_not_length_two():
    pairs, err = parse_clean_corrupt_pairs_json('[["a", "b", "c"]]')
    assert pairs is None
    assert "Entry 0" in err


def test_parse_pairs_non_string():
    pairs, err = parse_clean_corrupt_pairs_json("[[1, 2]]")
    assert pairs is None
    assert "strings" in err.lower()
