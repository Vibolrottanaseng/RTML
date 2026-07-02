import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.cyclegan import create_cyclegan_models
from utils.visualization import save_loss_plot


def train_cyclegan(
    dark_loader,
    blonde_loader,
    device: torch.device,
    epochs: int = 20,
    learning_rate: float = 2e-4,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 5.0,
    checkpoint_path: str = "saved/cyclegan_celeba.pt",
):
    (
        generator_dark_to_blonde,
        generator_blonde_to_dark,
        discriminator_dark,
        discriminator_blonde,
    ) = create_cyclegan_models(device)

    generator_optimizer = torch.optim.Adam(
        itertools.chain(
            generator_dark_to_blonde.parameters(),
            generator_blonde_to_dark.parameters(),
        ),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    discriminator_dark_optimizer = torch.optim.Adam(
        discriminator_dark.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    discriminator_blonde_optimizer = torch.optim.Adam(
        discriminator_blonde.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    adversarial_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    identity_criterion = nn.L1Loss()

    generator_losses = []
    discriminator_losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()

        generator_epoch_losses = []
        discriminator_epoch_losses = []

        batches = zip(dark_loader, blonde_loader)

        total_batches = min(
            len(dark_loader),
            len(blonde_loader),
        )

        progress = tqdm(
            batches,
            total=total_batches,
            desc=f"CycleGAN epoch {epoch + 1}/{epochs}",
        )

        for dark_batch, blonde_batch in progress:
            real_dark = dark_batch[0].to(device)
            real_blonde = blonde_batch[0].to(device)

            # -------------------------
            # Train both generators
            # -------------------------
            generator_optimizer.zero_grad()

            fake_blonde = generator_dark_to_blonde(
                real_dark
            )

            fake_dark = generator_blonde_to_dark(
                real_blonde
            )

            prediction_fake_blonde = (
                discriminator_blonde(fake_blonde)
            )

            prediction_fake_dark = (
                discriminator_dark(fake_dark)
            )

            valid_blonde = torch.ones_like(
                prediction_fake_blonde
            )

            valid_dark = torch.ones_like(
                prediction_fake_dark
            )

            adversarial_loss = (
                adversarial_criterion(
                    prediction_fake_blonde,
                    valid_blonde,
                )
                + adversarial_criterion(
                    prediction_fake_dark,
                    valid_dark,
                )
            )

            reconstructed_dark = (
                generator_blonde_to_dark(fake_blonde)
            )

            reconstructed_blonde = (
                generator_dark_to_blonde(fake_dark)
            )

            cycle_loss = (
                cycle_criterion(
                    reconstructed_dark,
                    real_dark,
                )
                + cycle_criterion(
                    reconstructed_blonde,
                    real_blonde,
                )
            )

            identity_dark = (
                generator_blonde_to_dark(real_dark)
            )

            identity_blonde = (
                generator_dark_to_blonde(real_blonde)
            )

            identity_loss = (
                identity_criterion(
                    identity_dark,
                    real_dark,
                )
                + identity_criterion(
                    identity_blonde,
                    real_blonde,
                )
            )

            generator_loss = (
                adversarial_loss
                + lambda_cycle * cycle_loss
                + lambda_identity * identity_loss
            )

            generator_loss.backward()
            generator_optimizer.step()

            # -------------------------
            # Train dark discriminator
            # -------------------------
            discriminator_dark_optimizer.zero_grad()

            prediction_real_dark = discriminator_dark(
                real_dark
            )

            real_dark_loss = adversarial_criterion(
                prediction_real_dark,
                torch.ones_like(prediction_real_dark),
            )

            prediction_fake_dark = discriminator_dark(
                fake_dark.detach()
            )

            fake_dark_loss = adversarial_criterion(
                prediction_fake_dark,
                torch.zeros_like(prediction_fake_dark),
            )

            discriminator_dark_loss = (
                real_dark_loss + fake_dark_loss
            ) * 0.5

            discriminator_dark_loss.backward()
            discriminator_dark_optimizer.step()

            # -------------------------
            # Train blonde discriminator
            # -------------------------
            discriminator_blonde_optimizer.zero_grad()

            prediction_real_blonde = (
                discriminator_blonde(real_blonde)
            )

            real_blonde_loss = adversarial_criterion(
                prediction_real_blonde,
                torch.ones_like(prediction_real_blonde),
            )

            prediction_fake_blonde = (
                discriminator_blonde(
                    fake_blonde.detach()
                )
            )

            fake_blonde_loss = adversarial_criterion(
                prediction_fake_blonde,
                torch.zeros_like(
                    prediction_fake_blonde
                ),
            )

            discriminator_blonde_loss = (
                real_blonde_loss + fake_blonde_loss
            ) * 0.5

            discriminator_blonde_loss.backward()
            discriminator_blonde_optimizer.step()

            discriminator_loss = (
                discriminator_dark_loss
                + discriminator_blonde_loss
            )

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

    Path(checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "generator_dark_to_blonde": (
                generator_dark_to_blonde.state_dict()
            ),
            "generator_blonde_to_dark": (
                generator_blonde_to_dark.state_dict()
            ),
            "discriminator_dark": (
                discriminator_dark.state_dict()
            ),
            "discriminator_blonde": (
                discriminator_blonde.state_dict()
            ),
            "lambda_cycle": lambda_cycle,
            "lambda_identity": lambda_identity,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "generator_losses": generator_losses,
            "discriminator_losses": discriminator_losses,
            "epoch_times": epoch_times,
        },
        checkpoint_path,
    )

    save_loss_plot(
        {
            "Generators": generator_losses,
            "Discriminators": discriminator_losses,
        },
        output_path="outputs/cyclegan/losses.png",
        title="CycleGAN training losses",
    )

    return (
        generator_dark_to_blonde,
        generator_blonde_to_dark,
    )