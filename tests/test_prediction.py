
import torch
from PIL import Image

from src.model import HandwrittenDigitCNN
from src.predict import predict_digit, preprocess_image


def test_preprocess_image():
    image = Image.new("RGB", (100, 100), "black")
    tensor = preprocess_image(image)

    assert tensor.shape == (1, 1, 28, 28)


def test_prediction_output():
    model = HandwrittenDigitCNN()
    image = Image.new("RGB", (100, 100), "black")

    result = predict_digit(image, model=model)

    assert 0 <= result["digit"] <= 9
    assert 0 <= result["confidence"] <= 1
    assert len(result["probabilities"]) == 10
