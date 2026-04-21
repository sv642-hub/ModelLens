from modellens.utils.token_display import prettify_subword_token, prettify_subword_tokens


def test_gpt2_style_space_marker():
    # GPT-2 BPE word-initial space prefix (U+0120) + letters
    raw = "\u0120capital"
    assert prettify_subword_token(raw) == " capital"


def test_sentencepiece_marker():
    raw = "\u2581hello"
    assert prettify_subword_token(raw) == " hello"


def test_list_mapping():
    assert prettify_subword_tokens(["The", "\u0120is"]) == ["The", " is"]
