import train


class FakeTokenizer:
    def encode(self, text):
        mapping = {
            "entrada": [10, 11],
            "resposta": [20, 21],
        }

        return mapping[text]


def test_teacher_forcing_shift(monkeypatch):
    monkeypatch.setattr(
        train,
        "MAX_LEN",
        8,
    )

    tokenizer = FakeTokenizer()

    src, tgt_input, tgt_labels = train.collate_batch(
        [("entrada", "resposta")],
        tokenizer,
    )

    assert src[0, :2].tolist() == [10, 11]

    assert tgt_input[0, :3].tolist() == [
        train.SOS_IDX,
        20,
        21,
    ]

    assert tgt_labels[0, :3].tolist() == [
        20,
        21,
        train.EOS_IDX,
    ]


def test_validation_split_preserves_categories():
    dataset = {
        "clear_to_rcfo": [
            (f"clear-{i}", f"target-{i}")
            for i in range(20)
        ],
        "vague_to_questions": [
            (f"vague-{i}", f"question-{i}")
            for i in range(20)
        ],
    }

    train_pairs, validation_pairs = train.split_dataset(
        dataset,
        validation_ratio=0.2,
        seed=42,
    )

    train_inputs = {
        item[0].split("-")[0]
        for item in train_pairs
    }

    validation_inputs = {
        item[0].split("-")[0]
        for item in validation_pairs
    }

    assert "clear" in train_inputs
    assert "vague" in train_inputs

    assert "clear" in validation_inputs
    assert "vague" in validation_inputs


def test_validation_split_is_reproducible():
    dataset = {
        "clear_to_rcfo": [
            (f"input-{i}", f"target-{i}")
            for i in range(30)
        ]
    }

    first = train.split_dataset(dataset, seed=123)
    second = train.split_dataset(dataset, seed=123)

    assert first == second


def test_validation_split_rejects_single_example():
    dataset = {
        "clear_to_rcfo": [
            ("only-input", "only-target")
        ]
    }

    try:
        train.split_dataset(dataset)
        assert False, "Era esperado ValueError."

    except ValueError:
        assert True