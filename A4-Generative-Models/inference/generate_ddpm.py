import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from models.ddpm import SimpleUNet
from utils.schedules import create_diffusion_schedule


@torch.no_grad()
def sample_ddpm(
    model,
    schedule,
    device,
    number_of_images=64,
    image_size=28,
    timesteps=1000,
):
    model.eval()

    images = torch.randn(
        number_of_images,
        1,
        image_size,
        image_size,
        device=device,
    )

    for timestep in reversed(range(timesteps)):
        timestep_batch = torch.full(
            (number_of_images,),
            timestep,
            device=device,
            dtype=torch.long,
        )

        predicted_noise = model(
            images,
            timestep_batch,
        )

        beta_t = schedule.betas[timestep]
        sqrt_one_minus_alpha_bar_t = (
            schedule.sqrt_one_minus_alpha_bar[timestep]
        )
        sqrt_recip_alpha_t = (
            schedule.sqrt_recip_alphas[timestep]
        )

        model_mean = sqrt_recip_alpha_t * (
            images
            - beta_t
            * predicted_noise
            / sqrt_one_minus_alpha_bar_t
        )

        if timestep > 0:
            noise = torch.randn_like(images)
            variance = schedule.posterior_variance[timestep]
            images = model_mean + torch.sqrt(variance) * noise
        else:
            images = model_mean

    return images.clamp(-1, 1)


def generate_from_checkpoint(
    checkpoint_path,
    output_path,
    device,
    number_of_images=64,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    schedule_name = checkpoint["schedule"]
    timesteps = checkpoint["timesteps"]

    model = SimpleUNet().to(device)
    model.load_state_dict(checkpoint["model"])

    schedule = create_diffusion_schedule(
        schedule_name=schedule_name,
        timesteps=timesteps,
        device=device,
    )

    generated_images = sample_ddpm(
        model=model,
        schedule=schedule,
        device=device,
        number_of_images=number_of_images,
        image_size=28,
        timesteps=timesteps,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid = make_grid(
        generated_images.cpu(),
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(
        grid.permute(1, 2, 0).squeeze(),
        cmap="gray",
    )
    plt.axis("off")
    plt.title(
        f"DDPM samples — {schedule_name} schedule"
    )
    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150,
    )

    plt.show()
    plt.close()

    print("Saved:", output_path)