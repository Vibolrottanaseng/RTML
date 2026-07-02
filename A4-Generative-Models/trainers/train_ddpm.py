import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.ddpm import SimpleUNet
from utils.schedules import create_diffusion_schedule
from utils.visualization import save_loss_plot


def q_sample(
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule,
) -> torch.Tensor:
    sqrt_alpha_bar = schedule.sqrt_alpha_bar[
        timesteps
    ][:, None, None, None]

    sqrt_one_minus_alpha_bar = (
        schedule.sqrt_one_minus_alpha_bar[
            timesteps
        ][:, None, None, None]
    )

    return (
        sqrt_alpha_bar * clean_images
        + sqrt_one_minus_alpha_bar * noise
    )


def train_ddpm(
    train_loader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 2e-4,
    timesteps: int = 1000,
    schedule_name: str = "linear",
    checkpoint_path: str = "saved/ddpm_mnist.pt",
):
    model = SimpleUNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    schedule = create_diffusion_schedule(
        schedule_name=schedule_name,
        timesteps=timesteps,
        device=device,
    )

    losses = []
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        epoch_losses = []

        progress = tqdm(
            train_loader,
            desc=f"DDPM epoch {epoch + 1}/{epochs}",
        )

        for clean_images, _ in progress:
            clean_images = clean_images.to(device)

            batch_size = clean_images.size(0)

            sampled_timesteps = torch.randint(
                low=0,
                high=timesteps,
                size=(batch_size,),
                device=device,
            )

            noise = torch.randn_like(clean_images)

            noisy_images = q_sample(
                clean_images=clean_images,
                timesteps=sampled_timesteps,
                noise=noise,
                schedule=schedule,
            )

            predicted_noise = model(
                noisy_images,
                sampled_timesteps,
            )

            loss = F.mse_loss(
                predicted_noise,
                noise,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {mean_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    Path(checkpoint_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "schedule": schedule_name,
            "timesteps": timesteps,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "losses": losses,
            "epoch_times": epoch_times,
        },
        checkpoint_path,
    )

    save_loss_plot(
        {"DDPM": losses},
        output_path=(
            f"outputs/ddpm/"
            f"{schedule_name}_losses.png"
        ),
        title=(
            f"DDPM training loss "
            f"({schedule_name} schedule)"
        ),
    )

    return model