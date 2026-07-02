# ============================================================
# Test CycleGAN with Your Own Face
# Keeps the full photo by resizing with padding
# ============================================================

import os
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from models.cyclegan import CycleGenerator


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 64

image_path = "my_face.jpg"
weights_path = "saved/cyclegan_lambda10.pt"
output_path = "outputs/cyclegan/my_face_result.png"

print("Using device:", device)


# ------------------------------------------------------------
# Helper: resize full image and add padding
# ------------------------------------------------------------

def resize_with_padding(
    image: Image.Image,
    size: int = 64,
    fill=(255, 255, 255),
) -> Image.Image:
    width, height = image.size

    scale = size / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    pad_right = size - new_width - pad_left
    pad_bottom = size - new_height - pad_top

    padded = TF.pad(
        resized,
        padding=[
            pad_left,
            pad_top,
            pad_right,
            pad_bottom,
        ],
        fill=fill,
    )

    return padded


# ------------------------------------------------------------
# Helper: convert normalized tensor for display
# ------------------------------------------------------------

def prepare_for_display(tensor: torch.Tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)

    return tensor.permute(1, 2, 0).numpy()


# ------------------------------------------------------------
# Main test function
# ------------------------------------------------------------

def test_own_face(
    image_path: str,
    weights_path: str,
    output_path: str,
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    # Load full original photo
    original_image = Image.open(
        image_path
    ).convert("RGB")

    # Keep the full image and add padding
    model_input_image = resize_with_padding(
        original_image,
        size=IMAGE_SIZE,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])

    image_tensor = (
        transform(model_input_image)
        .unsqueeze(0)
        .to(device)
    )

    print("Original size:", original_image.size)
    print("Model input size:", model_input_image.size)
    print("Input tensor shape:", image_tensor.shape)

    # Create the two generators
    G_d2b = CycleGenerator().to(device)
    G_b2d = CycleGenerator().to(device)

    # Load the checkpoint
    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    print("Checkpoint keys:", checkpoint.keys())

    # Support both your current checkpoint names
    # and the shorter names from the example.
    if "generator_dark_to_blonde" in checkpoint:
        G_d2b.load_state_dict(
            checkpoint["generator_dark_to_blonde"]
        )

        G_b2d.load_state_dict(
            checkpoint["generator_blonde_to_dark"]
        )

    elif "G_d2b" in checkpoint:
        G_d2b.load_state_dict(
            checkpoint["G_d2b"]
        )

        G_b2d.load_state_dict(
            checkpoint["G_b2d"]
        )

    elif "G" in checkpoint:
        G_d2b.load_state_dict(
            checkpoint["G"]
        )

        G_b2d.load_state_dict(
            checkpoint["F"]
        )

    else:
        raise KeyError(
            "Could not find CycleGAN generator keys "
            "inside the checkpoint."
        )

    G_d2b.eval()
    G_b2d.eval()

    # Generate translations
    with torch.no_grad():
        dark_to_blonde = G_d2b(image_tensor)
        blonde_to_dark = G_b2d(image_tensor)

    # Prepare generated images for display
    dark_to_blonde_display = prepare_for_display(
        dark_to_blonde
    )

    blonde_to_dark_display = prepare_for_display(
        blonde_to_dark
    )

    # Create output folder
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    # Plot the full original photo and both generated images
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Full Photo")
    axes[0].axis("off")

    axes[1].imshow(dark_to_blonde_display)
    axes[1].set_title("Dark → Blonde")
    axes[1].axis("off")

    axes[2].imshow(blonde_to_dark_display)
    axes[2].set_title("Blonde → Dark")
    axes[2].axis("off")

    plt.suptitle(
        "CycleGAN — Your Face",
        fontsize=14,
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


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

test_own_face(
    image_path=image_path,
    weights_path=weights_path,
    output_path=output_path,
)