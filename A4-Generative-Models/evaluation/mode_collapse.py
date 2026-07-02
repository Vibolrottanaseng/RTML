import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from evaluation.mnist_classifier import MNISTClassifier
from models.gan import Generator


def evaluate_mode_distribution(
    gan_checkpoint_path: str,
    classifier_checkpoint_path: str,
    device: torch.device,
    output_name: str,
    number_of_samples: int = 1000,
    batch_size: int = 100,
):
    gan_checkpoint = torch.load(
        gan_checkpoint_path,
        map_location=device,
    )

    classifier_checkpoint = torch.load(
        classifier_checkpoint_path,
        map_location=device,
    )

    z_dim = gan_checkpoint.get("z_dim", 100)

    generator = Generator(
        z_dim=z_dim
    ).to(device)

    generator.load_state_dict(
        gan_checkpoint["generator"]
    )

    classifier = MNISTClassifier().to(device)

    classifier.load_state_dict(
        classifier_checkpoint["model"]
    )

    generator.eval()
    classifier.eval()

    all_predictions = []

    generated_count = 0

    with torch.no_grad():
        while generated_count < number_of_samples:
            current_batch_size = min(
                batch_size,
                number_of_samples - generated_count,
            )

            noise = torch.randn(
                current_batch_size,
                z_dim,
                device=device,
            )

            generated_images = generator(noise)

            generated_images = generated_images.view(
                current_batch_size,
                1,
                28,
                28,
            )

            logits = classifier(generated_images)

            predictions = logits.argmax(dim=1)

            all_predictions.append(
                predictions.cpu()
            )

            generated_count += current_batch_size

    all_predictions = torch.cat(all_predictions)

    counts = torch.bincount(
        all_predictions,
        minlength=10,
    )

    counts_list = counts.tolist()

    output_directory = Path("outputs/gan")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / f"{output_name}_mode_counts.json"
    )

    with open(json_path, "w") as file:
        json.dump(
            {
                "checkpoint": gan_checkpoint_path,
                "number_of_samples": number_of_samples,
                "digit_counts": {
                    str(digit): count
                    for digit, count in enumerate(counts_list)
                },
            },
            file,
            indent=4,
        )

    plot_path = (
        output_directory
        / f"{output_name}_mode_distribution.png"
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(10),
        counts_list,
    )

    plt.xticks(range(10))
    plt.xlabel("Predicted digit")
    plt.ylabel("Number of generated samples")
    plt.title(
        f"GAN mode distribution: {output_name}"
    )

    plt.tight_layout()
    plt.savefig(
        plot_path,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nMode counts for {output_name}:")

    for digit, count in enumerate(counts_list):
        print(f"Digit {digit}: {count}")

    print(f"\nSaved counts to: {json_path}")
    print(f"Saved plot to: {plot_path}")

    return counts_list