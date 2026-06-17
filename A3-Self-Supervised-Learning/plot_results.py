import os
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

from models import MAE, build_dino_model
from evaluate import (
    linear_eval_dino,
    linear_eval_mae,
    visualize_mae_reconstruction,
    plot_tsne_dino_mae,
)
from utils import get_device, ensure_dirs, CLASSES, CIFAR_MEAN, CIFAR_STD, MAE_TEST_TF


os.makedirs("figures", exist_ok=True)


# ============================================================
# 1. Loss curves: DINO and MAE
# ============================================================

def load_stats(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_loss_curves():
    files = {
        "DINO default": "saved/dino_stats.json",
        "DINO no centering": "saved/dino_no_centering_stats.json",
        "DINO no local crops": "saved/dino_no_local_stats.json",
        "MAE mask 0.25": "saved/mae_encoder_mask025_stats.json",
        "MAE mask 0.50": "saved/mae_encoder_mask05_stats.json",
        "MAE mask 0.75": "saved/mae_encoder_mask075_stats.json",
    }

    plt.figure(figsize=(9, 5))

    for name, path in files.items():
        if os.path.exists(path):
            stats = load_stats(path)
            losses = stats["losses"]
            plt.plot(range(1, len(losses) + 1), losses, marker="o", label=name)
        else:
            print(f"Missing file: {path}")

    plt.title("Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/loss_curves_dino_mae.png", dpi=300)
    plt.show()


# ============================================================
# 2. MAE reconstruction grid
# original / masked / reconstructed
# ============================================================

def plot_mae_reconstruction(device):
    visualize_mae_reconstruction(
        device=device,
        weights="saved/mae_encoder_mask075.pt",
        mask_ratio=0.75,
        output_path="figures/mae_reconstruction_grid.png"
    )


# ============================================================
# 3. DINO attention map grid
# 10 images × all heads
# ============================================================

def denormalize_cifar(x):
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1)


def get_last_selfattention(vit, x):
    """
    Extract last-block self-attention from timm ViT.
    Returns attention with shape: [B, num_heads, tokens, tokens]
    """
    vit.eval()

    # Patch embedding
    x = vit.patch_embed(x)

    # Add class token
    cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)

    # Add positional embedding
    x = x + vit.pos_embed
    x = vit.pos_drop(x)

    # Forward through all blocks except the last
    for block in vit.blocks[:-1]:
        x = block(x)

    # Manually compute attention from the last block
    last_block = vit.blocks[-1]
    norm_x = last_block.norm1(x)
    attn_module = last_block.attn

    B, N, C = norm_x.shape
    qkv = attn_module.qkv(norm_x)
    qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)

    return attn


def plot_dino_attention_grid(device):
    vit, _ = build_dino_model()
    ckpt = torch.load("saved/dino.pt", map_location=device)
    vit.load_state_dict(ckpt["student_vit"])
    vit = vit.to(device)
    vit.eval()

    test_set = torchvision.datasets.CIFAR10(
        "./data",
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]),
        download=True,
    )

    loader = DataLoader(test_set, batch_size=10, shuffle=True)
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        attn = get_last_selfattention(vit, imgs)

    # CLS attention to patch tokens
    cls_attn = attn[:, :, 0, 1:]  # [B, heads, patches]

    B, H, P = cls_attn.shape
    grid_size = int(math.sqrt(P))

    cls_attn = cls_attn.reshape(B, H, grid_size, grid_size)

    # Upsample attention maps to image size
    cls_attn = F.interpolate(
        cls_attn,
        size=(32, 32),
        mode="bilinear",
        align_corners=False
    )

    fig, axes = plt.subplots(B, H + 1, figsize=(2 * (H + 1), 2 * B))

    for i in range(B):
        img = denormalize_cifar(imgs[i]).permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(CLASSES[labels[i].item()], fontsize=8)
        axes[i, 0].axis("off")

        for h in range(H):
            axes[i, h + 1].imshow(img)
            axes[i, h + 1].imshow(
                cls_attn[i, h].cpu().numpy(),
                cmap="jet",
                alpha=0.55
            )
            axes[i, h + 1].set_title(f"Head {h + 1}", fontsize=8)
            axes[i, h + 1].axis("off")

    plt.suptitle("DINO Attention Maps: 10 Images × All Heads", fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/dino_attention_grid.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 4. t-SNE comparison: DINO vs MAE
# ============================================================

def plot_tsne_comparison(device):
    print("Running DINO linear evaluation to collect embeddings...")
    dino_acc, dino_embeddings, dino_labels = linear_eval_dino(
        device=device,
        weights="saved/dino.pt",
        epochs=5,
        batch_size=256
    )

    print("Running MAE linear evaluation to collect embeddings...")
    mae_acc, mae_embeddings, mae_labels = linear_eval_mae(
        device=device,
        weights="saved/mae_encoder_mask075.pt",
        mask_ratio=0.75,
        epochs=5,
        batch_size=256
    )

    plot_tsne_dino_mae(
        dino_embeddings,
        dino_labels,
        mae_embeddings,
        mae_labels,
        output_path="figures/tsne_dino_vs_mae.png"
    )

    print(f"DINO Linear Eval Accuracy: {dino_acc:.2f}%")
    print(f"MAE Linear Eval Accuracy: {mae_acc:.2f}%")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ensure_dirs()
    device = get_device()
    print(f"Using device: {device}")

    plot_loss_curves()
    plot_mae_reconstruction(device)
    plot_dino_attention_grid(device)
    plot_tsne_comparison(device)