
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


class MNISTClassifier(nn.Module):
    """
    CNN classifier for MNIST digits.

    Input shape:
        batch_size × 1 × 28 × 28

    Output shape:
        batch_size × 10
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Input: 1 × 28 × 28
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Output: 32 × 14 × 14
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Output: 64 × 7 × 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_seed(seed=42):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_accuracy(model, dataloader, device):
    """Calculate classifier accuracy."""

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def train_classifier(args):
    """Train the MNIST classifier."""

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Classifier inputs are kept in the range [0, 1].
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = MNISTClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    os.makedirs("saved", exist_ok=True)

    best_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/"
                f"{args.epochs}"
            ),
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_loss = (
            running_loss / len(train_loader)
        )

        test_accuracy = calculate_accuracy(
            model,
            test_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {average_loss:.4f} | "
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "test_accuracy": test_accuracy,
                "epoch": epoch + 1,
                "learning_rate": args.learning_rate,
            }

            torch.save(
                checkpoint,
                "saved/mnist_classifier.pt",
            )

            print(
                "Saved best model to "
                "saved/mnist_classifier.pt"
            )

    print(
        "\nTraining completed."
    )

    print(
        f"Best test accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train a CNN classifier "
            "on the MNIST dataset."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    train_classifier(args)


if __name__ == "__main__":
    main()
