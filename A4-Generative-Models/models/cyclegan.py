import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CycleGenerator(nn.Module):
    """ResNet generator for 64x64 RGB images."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        ngf: int = 64,
        num_residual_blocks: int = 6,
    ):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                ngf,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                ngf * 2,
                ngf * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(ngf * 4))

        layers.extend([
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, in_channels: int = 3, ndf: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                ndf,
                ndf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                ndf * 2,
                ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                ndf * 4,
                ndf * 8,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                ndf * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_cyclegan_models(device: torch.device):
    """Create both generators and both discriminators."""

    generator_dark_to_blonde = CycleGenerator().to(device)
    generator_blonde_to_dark = CycleGenerator().to(device)

    discriminator_dark = PatchDiscriminator().to(device)
    discriminator_blonde = PatchDiscriminator().to(device)

    return (
        generator_dark_to_blonde,
        generator_blonde_to_dark,
        discriminator_dark,
        discriminator_blonde,
    )