from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from evaluation.mnist_classifier import MNISTClassifier


def train_mnist_classifier(
    train_loader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "saved/mnist_classifier.pt",
):
    model = MNISTClassifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Classifier epoch {epoch + 1}/{epochs}",
        )

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        average_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    Path(checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "epochs": epochs,
        },
        checkpoint_path,
    )

    print(f"Classifier saved to: {checkpoint_path}")

    return model