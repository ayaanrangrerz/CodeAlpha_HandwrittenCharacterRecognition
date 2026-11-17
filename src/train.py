
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import HandwrittenDigitCNN
from src.dataset import get_data_loaders


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Configuration
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    print(f"Using device: {DEVICE}")
    print("Loading MNIST dataset...")

    train_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE
    )

    model = HandwrittenDigitCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Accuracy: {train_accuracy:.2f}% "
            f"| Test Accuracy: {test_accuracy:.2f}%"
        )

    model_path = MODEL_DIR / "handwritten_digit_cnn.pth"

    torch.save(model.state_dict(), model_path)

    with open(REPORT_DIR / "training_history.json", "w") as file:
        json.dump(history, file, indent=4)

    final_metrics = {
        "model": "HandwrittenDigitCNN",
        "dataset": "MNIST",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_test_loss": history[-1]["test_loss"]
    }

    with open(REPORT_DIR / "metrics.json", "w") as file:
        json.dump(final_metrics, file, indent=4)

    print("\nTraining completed successfully!")
    print(f"Model saved: {model_path}")
    print(f"Final test accuracy: {history[-1]['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
