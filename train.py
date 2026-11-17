
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.model import HandwrittenDigitCNN
from src.dataset import get_data_loaders

SEED = 42
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


def main():
    print("Device:", DEVICE)
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
        total_loss = 0
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

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100 * correct / total

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    torch.save(
        model.state_dict(),
        "models/handwritten_digit_cnn.pth"
    )

    with open("reports/training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    with open("reports/metrics.json", "w") as f:
        json.dump({
            "model": "HandwrittenDigitCNN",
            "dataset": "MNIST",
            "epochs": EPOCHS,
            "test_accuracy": history[-1]["test_accuracy"]
        }, f, indent=4)

    print("\nTraining completed successfully!")
    print("Model saved: models/handwritten_digit_cnn.pth")
    print(f"Final accuracy: {history[-1]['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
