import io

import torch

from model import InspectorTransformer


def test_transformer_output_shape():
    vocab_size = 50

    model = InspectorTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_len=16,
    )

    src = torch.randint(0, vocab_size, (2, 8))
    tgt = torch.randint(0, vocab_size, (2, 6))

    model.eval()

    with torch.no_grad():
        output = model(src, tgt)

    assert output.shape == (2, 6, vocab_size)


def test_checkpoint_round_trip():
    config = {
        "vocab_size": 50,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 1,
        "d_ff": 64,
        "max_len": 16,
        "pad_idx": 0,
    }

    original_model = InspectorTransformer(**config)

    checkpoint = {
        "format_version": 2,
        "model_state_dict": original_model.state_dict(),
        "model_config": config,
    }

    buffer = io.BytesIO()

    torch.save(
        checkpoint,
        buffer,
    )

    buffer.seek(0)

    loaded_checkpoint = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    restored_model = InspectorTransformer(
        **loaded_checkpoint["model_config"]
    )

    restored_model.load_state_dict(
        loaded_checkpoint["model_state_dict"]
    )

    assert (
        restored_model.encoder.embedding.weight.shape
        == original_model.encoder.embedding.weight.shape
    )