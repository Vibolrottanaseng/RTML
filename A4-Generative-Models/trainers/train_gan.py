import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.gan import Discriminator, Generator
from utils.visualization import save_image_grid, save_loss_plot


def train_gan(
    train_loader,
    device: torch.device,
    epochs: int = 20,
    z_dim: int = 100,
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    checkpoint_path: str = "saved/gan_mnist.pt",
):
    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_lr,
        betas=(0.5, 0.999),
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=(0.5, 0.999),
    )

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(
        64,
        z_dim,
        device=device,
    )

    generator_losses = []
    discriminator_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()

        generator_epoch_losses = []
        discriminator_epoch_losses = []

        progress = tqdm(
            train_loader,
            desc=f"GAN epoch {epoch + 1}/{epochs}",
        )

        for real_images, _ in progress:
            batch_size = real_images.size(0)

            real_images = real_images.view(
                batch_size,
                -1,
            ).to(device)

            real_labels = torch.ones(
                batch_size,
                1,
                device=device,
            )

            fake_labels = torch.zeros(
                batch_size,
                1,
                device=device,
            )

            # Train discriminator
            noise = torch.randn(
                batch_size,
                z_dim,
                device=device,
            )

            fake_images = generator(noise).detach()

            real_loss = criterion(
                discriminator(real_images),
                real_labels,
            )

            fake_loss = criterion(
                discriminator(fake_images),
                fake_labels,
            )

            discriminator_loss = real_loss + fake_loss

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train generator
            noise = torch.randn(
                batch_size,
                z_dim,
                device=device,
            )

            generated_images = generator(noise)

            generator_loss = criterion(
                discriminator(generated_images),
                real_labels,
            )

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            generator_epoch_losses.append(
                generator_loss.item()
            )

            discriminator_epoch_losses.append(
                discriminator_loss.item()
            )

        generator_mean = float(
            np.mean(generator_epoch_losses)
        )

        discriminator_mean = float(
            np.mean(discriminator_epoch_losses)
        )

        generator_losses.append(generator_mean)
        discriminator_losses.append(discriminator_mean)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"G: {generator_mean:.4f} | "
            f"D: {discriminator_mean:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            generator.eval()

            with torch.no_grad():
                samples = generator(fixed_noise).view(
                    -1,
                    1,
                    28,
                    28,
                )

            save_image_grid(
                samples,
                output_path=(
                    f"outputs/gan/"
                    f"epoch_{epoch + 1:03d}.png"
                ),
                title=f"GAN epoch {epoch + 1}",
            )

            generator.train()

    Path(checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": (
                generator_optimizer.state_dict()
            ),
            "discriminator_optimizer": (
                discriminator_optimizer.state_dict()
            ),
            "epochs": epochs,
            "z_dim": z_dim,
            "generator_lr": generator_lr,
            "discriminator_lr": discriminator_lr,
            "generator_losses": generator_losses,
            "discriminator_losses": discriminator_losses,
            "epoch_times": epoch_times,
        },
        checkpoint_path,
    )

    save_loss_plot(
        {
            "Generator": generator_losses,
            "Discriminator": discriminator_losses,
        },
        output_path="outputs/gan/losses.png",
        title="GAN training losses",
    )

    return generator, discriminator