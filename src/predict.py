
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from src.model import HandwrittenDigitCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/handwritten_digit_cnn.pth")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python train.py first."
        )

    model = HandwrittenDigitCNN().to(DEVICE)
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_image(image):
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return transform(image).unsqueeze(0)


def predict_digit(image, model=None):
    if model is None:
        model = load_model()

    tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_digit = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_digit].item())

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities.cpu().tolist()
    }
