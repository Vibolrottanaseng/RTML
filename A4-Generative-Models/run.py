
import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm
from cyclegan_module import train_cyclegan

# ============================================================
# General utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# GAN model architecture
# ============================================================

class Generator(nn.Module):
    """
    Fully connected generator.

    Input:
        Random noise of shape [batch_size, z_dim]

    Output:
        Flattened MNIST images of shape [batch_size, 784]
    """

    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Fully connected discriminator.

    Input:
        Flattened image of shape [batch_size, 784]

    Output:
        Real/fake probability of shape [batch_size, 1]
    """

    def __init__(self, img_dim=784):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Dataset
# ============================================================

def get_mnist_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(f"MNIST training images: {len(dataset)}")

    return loader


# ============================================================
# Save generated image grid
# ============================================================

def save_generated_grid(
    generator,
    noise,
    output_path,
    title=None
):
    generator.eval()

    with torch.no_grad():
        fake_images = generator(noise)
        fake_images = fake_images.view(-1, 1, 28, 28).cpu()

    grid = torchvision.utils.make_grid(
        fake_images,
        nrow=8,
        normalize=True
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    generator.train()

    print(f"Saved generated samples to: {output_path}")


# ============================================================
# Save training curves
# ============================================================

def save_training_curves(
    generator_losses,
    discriminator_losses,
    epoch_times,
    output_path
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        generator_losses,
        label="Generator Loss"
    )

    axes[0].plot(
        discriminator_losses,
        label="Discriminator Loss"
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("GAN Training Losses")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        epoch_times,
        marker="o"
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title(
        f"Epoch Time, average: {np.mean(epoch_times):.1f}s"
    )
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved training curves to: {output_path}")


# ============================================================
# GAN training
# ============================================================

def train_gan(
    epochs,
    device,
    collapse=False,
    batch_size=128,
    z_dim=100
):
    os.makedirs("saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader = get_mnist_loader(batch_size=batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    generator_lr = 2e-4

    # Exercise 1b:
    # Use 6e-4 to intentionally make the discriminator stronger.
    discriminator_lr = 6e-4 if collapse else 2e-4

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=generator_lr,
        betas=(0.5, 0.999)
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=(0.5, 0.999)
    )

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(
        64,
        z_dim,
        device=device
    )

    generator_losses = []
    discriminator_losses = []
    epoch_times = []

    print()
    print("=" * 60)
    print("Starting GAN training")
    print(f"Generator learning rate:     {generator_lr}")
    print(f"Discriminator learning rate: {discriminator_lr}")
    print(f"Collapse experiment:         {collapse}")
    print("=" * 60)

    for epoch in range(epochs):
        start_time = time.time()

        epoch_generator_losses = []
        epoch_discriminator_losses = []

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for real_images, _ in progress_bar:
            current_batch_size = real_images.size(0)

            real_images = real_images.view(
                current_batch_size,
                -1
            ).to(device)

            real_labels = torch.ones(
                current_batch_size,
                1,
                device=device
            )

            fake_labels = torch.zeros(
                current_batch_size,
                1,
                device=device
            )

            # ----------------------------------------------
            # Train discriminator
            # ----------------------------------------------

            noise = torch.randn(
                current_batch_size,
                z_dim,
                device=device
            )

            fake_images = generator(noise).detach()

            real_predictions = discriminator(real_images)
            fake_predictions = discriminator(fake_images)

            real_loss = criterion(
                real_predictions,
                real_labels
            )

            fake_loss = criterion(
                fake_predictions,
                fake_labels
            )

            discriminator_loss = real_loss + fake_loss

            optimizer_d.zero_grad()
            discriminator_loss.backward()
            optimizer_d.step()

            # ----------------------------------------------
            # Train generator
            # ----------------------------------------------

            noise = torch.randn(
                current_batch_size,
                z_dim,
                device=device
            )

            generated_images = generator(noise)
            predictions = discriminator(generated_images)

            generator_loss = criterion(
                predictions,
                real_labels
            )

            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

            epoch_generator_losses.append(
                generator_loss.item()
            )

            epoch_discriminator_losses.append(
                discriminator_loss.item()
            )

            progress_bar.set_postfix({
                "G": f"{generator_loss.item():.3f}",
                "D": f"{discriminator_loss.item():.3f}"
            })

        epoch_time = time.time() - start_time

        average_generator_loss = np.mean(
            epoch_generator_losses
        )

        average_discriminator_loss = np.mean(
            epoch_discriminator_losses
        )

        generator_losses.append(average_generator_loss)
        discriminator_losses.append(average_discriminator_loss)
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"G loss: {average_generator_loss:.4f} | "
            f"D loss: {average_discriminator_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save a sample grid every five epochs.
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            experiment_name = (
                "collapse" if collapse else "normal"
            )

            sample_path = (
                f"results/gan_{experiment_name}_"
                f"epoch_{epoch + 1}.png"
            )

            save_generated_grid(
                generator,
                fixed_noise,
                sample_path,
                title=f"GAN samples — epoch {epoch + 1}"
            )

    # ------------------------------------------------------
    # Save trained model
    # ------------------------------------------------------

    if collapse:
        weights_path = "saved/gan_mnist_collapse.pt"
        final_grid_path = "results/gan_collapse_samples.png"
        curves_path = "results/gan_collapse_training.png"
    else:
        weights_path = "saved/gan_mnist.pt"
        final_grid_path = "results/gan_samples.png"
        curves_path = "results/gan_training.png"

    checkpoint = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "z_dim": z_dim,
        "epochs": epochs,
        "generator_lr": generator_lr,
        "discriminator_lr": discriminator_lr,
        "collapse": collapse,
        "generator_losses": generator_losses,
        "discriminator_losses": discriminator_losses,
        "epoch_times": epoch_times
    }

    torch.save(checkpoint, weights_path)

    save_generated_grid(
        generator,
        fixed_noise,
        final_grid_path,
        title="Final GAN Generated Samples"
    )

    save_training_curves(
        generator_losses,
        discriminator_losses,
        epoch_times,
        curves_path
    )

    print()
    print("GAN training completed.")
    print(f"Weights saved to: {weights_path}")

    return generator


# ============================================================
# Generate from saved GAN
# ============================================================

def generate_gan_samples(
    weights_path,
    number_of_samples,
    device
):
    checkpoint = torch.load(
        weights_path,
        map_location=device
    )

    z_dim = checkpoint.get("z_dim", 100)

    generator = Generator(z_dim=z_dim).to(device)

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    generator.eval()

    noise = torch.randn(
        number_of_samples,
        z_dim,
        device=device
    )

    output_path = "results/gan_generated.png"

    save_generated_grid(
        generator,
        noise[:64],
        output_path,
        title="Generated MNIST Samples"
    )


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="A4 Generative Models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gan", "cyclegan", "ddpm"]
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "celeba"],
        default=None
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None
    )

    parser.add_argument(
        "--test-image",
        type=str,
        default=None
    )

    parser.add_argument(
        "--schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"]
    )

    parser.add_argument(
        "--n",
        type=int,
        default=64
    )

    parser.add_argument(
        "--train",
        action="store_true"
    )

    parser.add_argument(
        "--generate",
        action="store_true"
    )

    parser.add_argument(
        "--collapse",
        action="store_true"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/content/data/celeba/img_align_celeba"
    )
    
    parser.add_argument(
        "--attribute-file",
        type=str,
        default="/content/data/celeba/list_attr_celeba.txt"
    )
    
    parser.add_argument(
        "--lambda-cycle",
        type=float,
        default=10.0
    )
    
    parser.add_argument(
        "--lambda-identity",
        type=float,
        default=5.0
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=5000
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(42)

    device = get_device()

    print("=" * 60)
    print("A4 Generative Models")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Dataset:  {args.dataset}")
    print(f"Epochs:   {args.epochs}")
    print(f"Device:   {device}")
    print("=" * 60)

    if args.model == "gan":
        if args.train:
            train_gan(
                epochs=args.epochs,
                device=device,
                collapse=args.collapse
            )

        elif args.generate:
            if args.weights is None:
                raise ValueError(
                    "--weights is required when using --generate"
                )

            generate_gan_samples(
                weights_path=args.weights,
                number_of_samples=args.n,
                device=device
            )

        else:
            print(
                "Use either --train or --generate for the GAN."
            )

    elif args.model == "cyclegan":
        if not args.train:
            print("Use --train to train CycleGAN.")
            return

    train_cyclegan(
        image_dir=args.image_dir,
        attribute_file=args.attribute_file,
        epochs=args.epochs,
        device=device,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
