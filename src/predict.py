"""Image preprocessing and CNN inference helpers."""
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from src.model import HandwrittenDigitCNN

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "handwritten_digit_cnn.pth"


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run python train.py first.")
    model = HandwrittenDigitCNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert a canvas image to the normalized MNIST input tensor."""
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return transform(image).unsqueeze(0)


def predict_digit(image: Image.Image, model=None):
    """Return predicted digit, confidence, and probabilities for classes 0-9."""
    model = model or load_model()
    device = next(model.parameters()).device
    tensor = preprocess_image(image).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    predicted_digit = int(probabilities.argmax().item())
    return {
        "digit": predicted_digit,
        "confidence": float(probabilities[predicted_digit].item()),
        "probabilities": probabilities.cpu().tolist(),
    }
