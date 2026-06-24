
import argparse
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the Generator architecture from run.py
from run import Generator


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MNIST classifier
# ============================================================

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# MNIST DataLoaders
# ============================================================

def get_mnist_loaders(batch_size=128):
    # Match the GAN output range: [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


# ============================================================
# Train classifier
# ============================================================

def train_classifier(
    classifier,
    train_loader,
    test_loader,
    device,
    epochs=3
):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=1e-3
    )

    print("\nTraining MNIST classifier...")

    for epoch in range(epochs):
        classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Classifier epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = classifier(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "accuracy": f"{100 * correct / total:.2f}%"
            })

        test_accuracy = evaluate_classifier(
            classifier,
            test_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}: "
            f"training accuracy={100 * correct / total:.2f}% | "
            f"test accuracy={test_accuracy:.2f}%"
        )

    return classifier


# ============================================================
# Evaluate classifier
# ============================================================

def evaluate_classifier(classifier, data_loader, device):
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = classifier(images)
            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total


# ============================================================
# Load or train classifier
# ============================================================

def load_or_train_classifier(device, classifier_path):
    classifier = MNISTClassifier().to(device)

    if os.path.exists(classifier_path):
        print(f"Loading classifier: {classifier_path}")

        state_dict = torch.load(
            classifier_path,
            map_location=device,
            weights_only=True
        )

        classifier.load_state_dict(state_dict)

        _, test_loader = get_mnist_loaders()

        test_accuracy = evaluate_classifier(
            classifier,
            test_loader,
            device
        )

        print(
            f"Classifier test accuracy: "
            f"{test_accuracy:.2f}%"
        )

        return classifier

    train_loader, test_loader = get_mnist_loaders()

    classifier = train_classifier(
        classifier=classifier,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=3
    )

    torch.save(
        classifier.state_dict(),
        classifier_path
    )

    print(f"Classifier saved to: {classifier_path}")

    return classifier


# ============================================================
# Load a trained GAN
# ============================================================

def load_generator(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"GAN checkpoint was not found: {weights_path}"
        )

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )

    z_dim = checkpoint.get("z_dim", 100)

    generator = Generator(
        z_dim=z_dim
    ).to(device)

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.eval()

    return generator, z_dim


# ============================================================
# Generate and classify GAN samples
# ============================================================

def classify_generated_samples(
    generator,
    classifier,
    z_dim,
    number_of_samples,
    device,
    batch_size=100
):
    counts = torch.zeros(
        10,
        dtype=torch.long
    )

    all_images = []
    all_predictions = []
    all_confidences = []

    generator.eval()
    classifier.eval()

    generated_so_far = 0

    with torch.no_grad():
        while generated_so_far < number_of_samples:
            current_batch_size = min(
                batch_size,
                number_of_samples - generated_so_far
            )

            noise = torch.randn(
                current_batch_size,
                z_dim,
                device=device
            )

            generated = generator(noise)

            generated = generated.view(
                current_batch_size,
                1,
                28,
                28
            )

            logits = classifier(generated)
            probabilities = torch.softmax(logits, dim=1)

            confidence, predictions = probabilities.max(dim=1)

            batch_counts = torch.bincount(
                predictions.cpu(),
                minlength=10
            )

            counts += batch_counts

            all_images.append(generated.cpu())
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidence.cpu())

            generated_so_far += current_batch_size

    images = torch.cat(all_images, dim=0)
    predictions = torch.cat(all_predictions, dim=0)
    confidences = torch.cat(all_confidences, dim=0)

    return counts.numpy(), images, predictions, confidences


# ============================================================
# Print count table
# ============================================================

def print_count_table(name, counts):
    print()
    print("=" * 74)
    print(name)
    print("=" * 74)

    digit_header = "Digit | " + " | ".join(
        str(digit) for digit in range(10)
    )

    separator = "-" * len(digit_header)

    count_row = "Count | " + " | ".join(
        str(int(count)) for count in counts
    )

    print(digit_header)
    print(separator)
    print(count_row)
    print(f"Total: {counts.sum()}")

    vanished_digits = np.where(counts == 0)[0].tolist()

    if vanished_digits:
        print(f"Vanished digits: {vanished_digits}")
    else:
        print("No digit has a count of exactly zero.")


# ============================================================
# Save sample grid with predicted labels
# ============================================================

def save_labeled_grid(
    images,
    predictions,
    confidences,
    output_path,
    title
):
    number_to_show = min(64, len(images))

    fig, axes = plt.subplots(
        8,
        8,
        figsize=(10, 11)
    )

    for index, axis in enumerate(axes.flat):
        if index < number_to_show:
            image = images[index].squeeze().numpy()

            axis.imshow(
                image,
                cmap="gray",
                vmin=-1,
                vmax=1
            )

            axis.set_title(
                f"{predictions[index].item()} "
                f"({confidences[index].item():.2f})",
                fontsize=7
            )

        axis.axis("off")

    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved labeled samples to: {output_path}")


# ============================================================
# Plot separate distribution
# ============================================================

def save_distribution_plot(
    counts,
    output_path,
    title
):
    digits = np.arange(10)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(digits, counts)

    plt.axhline(
        y=100,
        linestyle="--",
        label="Perfectly even count = 100"
    )

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(count)),
            ha="center",
            fontsize=9
        )

    plt.xticks(digits)
    plt.xlabel("Predicted digit")
    plt.ylabel("Count out of 1,000")
    plt.title(title)
    plt.ylim(0, max(counts.max() + 60, 150))
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved distribution plot to: {output_path}")


# ============================================================
# Plot normal versus collapse
# ============================================================

def save_comparison_plot(
    normal_counts,
    collapse_counts,
    output_path
):
    digits = np.arange(10)
    width = 0.38

    plt.figure(figsize=(11, 6))

    plt.bar(
        digits - width / 2,
        normal_counts,
        width,
        label="Normal GAN"
    )

    plt.bar(
        digits + width / 2,
        collapse_counts,
        width,
        label="High discriminator LR"
    )

    plt.axhline(
        y=100,
        linestyle="--",
        label="Perfectly even count = 100"
    )

    plt.xticks(digits)
    plt.xlabel("Predicted digit")
    plt.ylabel("Count out of 1,000")
    plt.title("GAN Digit Distribution: Normal vs Mode-Collapse Experiment")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved comparison plot to: {output_path}")


# ============================================================
# Distribution metrics
# ============================================================

def calculate_distribution_metrics(counts):
    probabilities = counts / counts.sum()

    entropy = -np.sum(
        probabilities[probabilities > 0]
        * np.log(probabilities[probabilities > 0])
    )

    normalized_entropy = entropy / np.log(10)

    expected_count = counts.sum() / 10

    mean_absolute_deviation = np.mean(
        np.abs(counts - expected_count)
    )

    covered_digits = int(np.sum(counts > 0))

    return {
        "covered_digits": covered_digits,
        "normalized_entropy": normalized_entropy,
        "mean_absolute_deviation": mean_absolute_deviation,
        "minimum_count": int(counts.min()),
        "maximum_count": int(counts.max())
    }


# ============================================================
# Save results as CSV
# ============================================================

def save_results_csv(
    normal_counts,
    collapse_counts,
    output_path
):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "Digit",
            "Normal GAN count",
            "Collapse GAN count"
        ])

        for digit in range(10):
            writer.writerow([
                digit,
                int(normal_counts[digit]),
                int(collapse_counts[digit])
            ])

        writer.writerow([
            "Total",
            int(normal_counts.sum()),
            int(collapse_counts.sum())
        ])

    print(f"Saved count table to: {output_path}")


# ============================================================
# Main evaluation
# ============================================================

def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs("saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print(f"Device: {device}")

    classifier_path = "saved/mnist_classifier.pt"

    classifier = load_or_train_classifier(
        device=device,
        classifier_path=classifier_path
    )

    print("\nLoading normal GAN...")
    normal_generator, normal_z_dim = load_generator(
        args.normal_weights,
        device
    )

    print("Generating and classifying normal GAN samples...")

    (
        normal_counts,
        normal_images,
        normal_predictions,
        normal_confidences
    ) = classify_generated_samples(
        generator=normal_generator,
        classifier=classifier,
        z_dim=normal_z_dim,
        number_of_samples=args.n,
        device=device
    )

    print("\nLoading collapse GAN...")
    collapse_generator, collapse_z_dim = load_generator(
        args.collapse_weights,
        device
    )

    print("Generating and classifying collapse GAN samples...")

    (
        collapse_counts,
        collapse_images,
        collapse_predictions,
        collapse_confidences
    ) = classify_generated_samples(
        generator=collapse_generator,
        classifier=classifier,
        z_dim=collapse_z_dim,
        number_of_samples=args.n,
        device=device
    )

    print_count_table(
        "Normal GAN distribution",
        normal_counts
    )

    print_count_table(
        "Mode-collapse experiment distribution",
        collapse_counts
    )

    normal_metrics = calculate_distribution_metrics(
        normal_counts
    )

    collapse_metrics = calculate_distribution_metrics(
        collapse_counts
    )

    print("\nNormal GAN metrics:")
    for key, value in normal_metrics.items():
        print(f"  {key}: {value}")

    print("\nCollapse GAN metrics:")
    for key, value in collapse_metrics.items():
        print(f"  {key}: {value}")

    save_distribution_plot(
        counts=normal_counts,
        output_path="results/gan_distribution.png",
        title="Normal GAN Digit Distribution"
    )

    save_distribution_plot(
        counts=collapse_counts,
        output_path="results/gan_collapse_distribution.png",
        title="GAN Distribution with Discriminator LR = 6e-4"
    )

    save_comparison_plot(
        normal_counts=normal_counts,
        collapse_counts=collapse_counts,
        output_path="results/gan_distribution_comparison.png"
    )

    save_labeled_grid(
        images=normal_images,
        predictions=normal_predictions,
        confidences=normal_confidences,
        output_path="results/gan_classified_samples.png",
        title="Normal GAN Samples with Classifier Predictions"
    )

    save_labeled_grid(
        images=collapse_images,
        predictions=collapse_predictions,
        confidences=collapse_confidences,
        output_path="results/gan_collapse_classified_samples.png",
        title="Collapse GAN Samples with Classifier Predictions"
    )

    save_results_csv(
        normal_counts=normal_counts,
        collapse_counts=collapse_counts,
        output_path="results/gan_digit_counts.csv"
    )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GAN mode collapse"
    )

    parser.add_argument(
        "--normal-weights",
        type=str,
        default="saved/gan_mnist.pt"
    )

    parser.add_argument(
        "--collapse-weights",
        type=str,
        default="saved/gan_mnist_collapse.pt"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()
    main(args)
