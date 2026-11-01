from tokenizer import BPETokenizer


def test_tokenizer_round_trip():
    tokenizer = BPETokenizer(vocab_size=64)

    sample = "Inspector AI creates structured prompts."

    tokenizer.train(
        [sample] * 10,
        max_base_chars=64,
    )

    encoded = tokenizer.encode(sample)

    decoded = tokenizer.decode(
        encoded,
        skip_special=True,
    )

    assert decoded == sample


def test_special_tokens_have_reserved_ids():
    from tokenizer import PAD_IDX, SOS_IDX, EOS_IDX

    assert PAD_IDX == 0
    assert SOS_IDX == 1
    assert EOS_IDX == 2