
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# CelebA dataset
# ============================================================

class CelebAHairDataset(Dataset):
    def __init__(
        self,
        image_dir,
        attribute_file,
        domain,
        image_size=64
    ):
        self.image_dir = image_dir
        self.domain = domain

        attributes = pd.read_csv(
            attribute_file,
            sep=r"\s+",
            skiprows=1
        )

        # CelebA values are -1 and 1.
        # Blonde_Hair=1 forms the blonde domain.
        # Black_Hair or Brown_Hair forms the dark-hair domain.
        if domain == "blonde":
            selected = attributes[
                attributes["Blond_Hair"] == 1
            ]
        elif domain == "dark":
            selected = attributes[
                (attributes["Black_Hair"] == 1)
                | (attributes["Brown_Hair"] == 1)
            ]
        else:
            raise ValueError(
                "domain must be 'dark' or 'blonde'"
            )

        

        self.filenames = selected.index.tolist()

        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

        print(
            f"{domain.capitalize()} hair images: "
            f"{len(self.filenames)}"
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image


def create_celeba_loaders(
    image_dir,
    attribute_file,
    batch_size=16,
    image_size=64
):
    dark_dataset = CelebAHairDataset(
        image_dir=image_dir,
        attribute_file=attribute_file,
        domain="dark",
        image_size=image_size,
    )

    blonde_dataset = CelebAHairDataset(
        image_dir=image_dir,
        attribute_file=attribute_file,
        domain="blonde",
        image_size=image_size,
   
    )

    dark_loader = DataLoader(
        dark_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    blonde_loader = DataLoader(
        blonde_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    return dark_loader, blonde_loader


# ============================================================
# CycleGAN architecture
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGenerator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=64,
        residual_blocks=6
    ):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                in_channels,
                features,
                kernel_size=7
            ),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features * 2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features * 2,
                features * 4,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True)
        ]

        for _ in range(residual_blocks):
            layers.append(
                ResidualBlock(features * 4)
            )

        layers.extend([
            nn.ConvTranspose2d(
                features * 4,
                features * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                features * 2,
                features,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(
                features,
                out_channels,
                kernel_size=7
            ),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        def block(in_features, out_features, normalize=True):
            layers = [
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ]

            if normalize:
                layers.append(
                    nn.InstanceNorm2d(out_features)
                )

            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )

            return layers

        self.model = nn.Sequential(
            *block(
                in_channels,
                features,
                normalize=False
            ),
            *block(features, features * 2),
            *block(features * 2, features * 4),

            nn.ZeroPad2d(1),
            nn.Conv2d(
                features * 4,
                1,
                kernel_size=4,
                padding=1
            )
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Visualization
# ============================================================

def denormalize(images):
    return (images * 0.5 + 0.5).clamp(0, 1)


def save_translation_grid(
    generator_g,
    generator_f,
    dark_images,
    blonde_images,
    output_path,
    title
):
    generator_g.eval()
    generator_f.eval()

    with torch.no_grad():
        fake_blonde = generator_g(dark_images).cpu()
        fake_dark = generator_f(blonde_images).cpu()

    dark_images = dark_images.cpu()
    blonde_images = blonde_images.cpu()

    groups = [
        dark_images,
        fake_blonde,
        blonde_images,
        fake_dark
    ]

    labels = [
        "Real dark",
        "Dark to blonde",
        "Real blonde",
        "Blonde to dark"
    ]

    figure, axes = plt.subplots(
        4,
        4,
        figsize=(10, 10)
    )

    for row in range(4):
        for column in range(4):
            image = denormalize(
                groups[row][column]
            )

            axes[row, column].imshow(
                image.permute(1, 2, 0)
            )
            axes[row, column].axis("off")

        axes[row, 0].set_ylabel(
            labels[row],
            fontsize=9
        )

    figure.suptitle(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    generator_g.train()
    generator_f.train()


# ============================================================
# CycleGAN training
# ============================================================

def train_cyclegan(
    image_dir,
    attribute_file,
    epochs,
    device,
    batch_size=16,
    lambda_cycle = 10.0,
    lambda_identity = 5.0
):
    os.makedirs("saved", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    dark_loader, blonde_loader = create_celeba_loaders(
        image_dir=image_dir,
        attribute_file=attribute_file,
        batch_size=batch_size,
        
    )

    generator_g = CycleGenerator().to(device)
    generator_f = CycleGenerator().to(device)

    discriminator_x = PatchDiscriminator().to(device)
    discriminator_y = PatchDiscriminator().to(device)

    generator_optimizer = torch.optim.Adam(
        list(generator_g.parameters())
        + list(generator_f.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999)
    )

    discriminator_optimizer = torch.optim.Adam(
        list(discriminator_x.parameters())
        + list(discriminator_y.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999)
    )

    adversarial_loss = nn.MSELoss()
    reconstruction_loss = nn.L1Loss()

    generator_losses = []
    discriminator_losses = []
    epoch_times = []

    fixed_dark = next(iter(dark_loader))[:4].to(device)
    fixed_blonde = next(iter(blonde_loader))[:4].to(device)

    for epoch in range(epochs):
        start_time = time.time()

        epoch_generator_losses = []
        epoch_discriminator_losses = []

        dark_iterator = iter(dark_loader)
        blonde_iterator = iter(blonde_loader)

        number_of_batches = min(
            len(dark_loader),
            len(blonde_loader)
        )

        progress = tqdm(
            range(number_of_batches),
            desc=f"CycleGAN epoch {epoch + 1}/{epochs}"
        )

        for _ in progress:
            real_x = next(dark_iterator).to(device)
            real_y = next(blonde_iterator).to(device)

            # ----------------------------------------------
            # Train generators
            # ----------------------------------------------

            generator_optimizer.zero_grad()

            fake_y = generator_g(real_x)
            fake_x = generator_f(real_y)

            cycle_x = generator_f(fake_y)
            cycle_y = generator_g(fake_x)

            identity_x = generator_f(real_x)
            identity_y = generator_g(real_y)

            real_y_labels = torch.ones_like(
                discriminator_y(fake_y)
            )

            real_x_labels = torch.ones_like(
                discriminator_x(fake_x)
            )

            generator_adversarial = (
                adversarial_loss(
                    discriminator_y(fake_y),
                    real_y_labels
                )
                + adversarial_loss(
                    discriminator_x(fake_x),
                    real_x_labels
                )
            )

            cycle_component = (
                reconstruction_loss(cycle_x, real_x)
                + reconstruction_loss(cycle_y, real_y)
            )

            identity_component = (
                reconstruction_loss(identity_x, real_x)
                + reconstruction_loss(identity_y, real_y)
            )

            generator_loss = (
                generator_adversarial
                + lambda_cycle * cycle_component
                + lambda_identity * identity_component
            )

            generator_loss.backward()
            generator_optimizer.step()

            # ----------------------------------------------
            # Train discriminators
            # ----------------------------------------------

            discriminator_optimizer.zero_grad()

            real_x_output = discriminator_x(real_x)
            fake_x_output = discriminator_x(
                fake_x.detach()
            )

            real_y_output = discriminator_y(real_y)
            fake_y_output = discriminator_y(
                fake_y.detach()
            )

            discriminator_x_loss = (
                adversarial_loss(
                    real_x_output,
                    torch.ones_like(real_x_output)
                )
                + adversarial_loss(
                    fake_x_output,
                    torch.zeros_like(fake_x_output)
                )
            )

            discriminator_y_loss = (
                adversarial_loss(
                    real_y_output,
                    torch.ones_like(real_y_output)
                )
                + adversarial_loss(
                    fake_y_output,
                    torch.zeros_like(fake_y_output)
                )
            )

            discriminator_loss = (
                discriminator_x_loss
                + discriminator_y_loss
            ) * 0.5

            discriminator_loss.backward()
            discriminator_optimizer.step()

            epoch_generator_losses.append(
                generator_loss.item()
            )

            epoch_discriminator_losses.append(
                discriminator_loss.item()
            )

            progress.set_postfix({
                "G": f"{generator_loss.item():.3f}",
                "D": f"{discriminator_loss.item():.3f}"
            })

        elapsed = time.time() - start_time

        average_generator_loss = np.mean(
            epoch_generator_losses
        )

        average_discriminator_loss = np.mean(
            epoch_discriminator_losses
        )

        generator_losses.append(
            average_generator_loss
        )

        discriminator_losses.append(
            average_discriminator_loss
        )

        epoch_times.append(elapsed)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"G: {average_generator_loss:.3f} | "
            f"D: {average_discriminator_loss:.3f} | "
            f"{elapsed:.1f}s"
        )

        if lambda_cycle == 0:
            experiment_name = "no_cycle"
        else:
            experiment_name = "default"

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            save_translation_grid(
                generator_g=generator_g,
                generator_f=generator_f,
                dark_images=fixed_dark,
                blonde_images=fixed_blonde,
                output_path=(
                    f"results/cyclegan_{experiment_name}_"
                    f"epoch_{epoch + 1}.png"
                ),
                title=(
                    f"CycleGAN {experiment_name} — "
                    f"epoch {epoch + 1}"
                )
            )

    if lambda_cycle == 0:
        checkpoint_path = (
            "saved/cyclegan_no_cycle.pt"
        )
        final_result_path = (
            "results/cyclegan_no_cycle.png"
        )
    else:
        checkpoint_path = (
            "saved/cyclegan_celeba.pt"
        )
        final_result_path = (
            "results/cyclegan_default.png"
        )

    checkpoint = {
        "G": generator_g.state_dict(),
        "F": generator_f.state_dict(),
        "D_X": discriminator_x.state_dict(),
        "D_Y": discriminator_y.state_dict(),
        "epochs": epochs,
        "lambda_cycle": lambda_cycle,
        "lambda_identity": lambda_identity,
        "generator_losses": generator_losses,
        "discriminator_losses": discriminator_losses,
        "epoch_times": epoch_times
    }

    torch.save(checkpoint, checkpoint_path)

    save_translation_grid(
        generator_g=generator_g,
        generator_f=generator_f,
        dark_images=fixed_dark,
        blonde_images=fixed_blonde,
        output_path=final_result_path,
        title=(
            f"CycleGAN results — "
            f"lambda_cycle={lambda_cycle}"
        )
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved results: {final_result_path}")
