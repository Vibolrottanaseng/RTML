from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def save_image_grid(
    images: torch.Tensor,
    output_path: str,
    title: str,
    nrow: int = 8,
) -> None:
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = images.detach().cpu()

    grid = make_grid(
        images,
        nrow=nrow,
        normalize=True,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()


def save_loss_plot(
    losses: dict,
    output_path: str,
    title: str,
) -> None:
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    for label, values in losses.items():
        plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()