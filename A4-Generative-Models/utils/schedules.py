from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bar: torch.Tensor
    sqrt_alpha_bar: torch.Tensor
    sqrt_one_minus_alpha_bar: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    return torch.linspace(
        beta_start,
        beta_end,
        timesteps,
    )


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    t = torch.linspace(
        0,
        timesteps,
        timesteps + 1,
    )

    alpha_bar = torch.cos(
        ((t / timesteps) + s)
        / (1 + s)
        * torch.pi
        * 0.5
    ) ** 2

    alpha_bar = alpha_bar / alpha_bar[0]

    betas = 1 - (
        alpha_bar[1:] / alpha_bar[:-1]
    )

    return torch.clamp(
        betas,
        0.0001,
        0.9999,
    )


def create_diffusion_schedule(
    schedule_name: str,
    timesteps: int,
    device: torch.device,
) -> DiffusionSchedule:
    if schedule_name == "linear":
        betas = linear_beta_schedule(timesteps)

    elif schedule_name == "cosine":
        betas = cosine_beta_schedule(timesteps)

    else:
        raise ValueError(
            f"Unknown schedule: {schedule_name}. "
            "Use 'linear' or 'cosine'."
        )

    betas = betas.to(device)

    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(
        1.0 - alpha_bar
    )

    sqrt_recip_alphas = torch.sqrt(
        1.0 / alphas
    )

    previous_alpha_bar = torch.nn.functional.pad(
        alpha_bar[:-1],
        (1, 0),
        value=1.0,
    )

    posterior_variance = (
        betas
        * (1.0 - previous_alpha_bar)
        / (1.0 - alpha_bar)
    )

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
        sqrt_alpha_bar=sqrt_alpha_bar,
        sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )