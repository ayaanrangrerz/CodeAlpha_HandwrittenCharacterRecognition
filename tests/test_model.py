import torch

from src.model import HandwrittenDigitCNN


def test_model_output_shape_and_finite_values():
    model = HandwrittenDigitCNN()
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(4, 1, 28, 28))
    assert output.shape == (4, 10)
    assert torch.isfinite(output).all()


def test_model_supports_single_image_batch():
    model = HandwrittenDigitCNN()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 28, 28))
    assert output.shape == (1, 10)
