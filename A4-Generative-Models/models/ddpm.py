import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2

        scale = math.log(10000) / max(half_dim - 1, 1)

        frequencies = torch.exp(
            torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            ) * -scale
        )

        embeddings = timesteps.float()[:, None] * frequencies[None, :]

        embeddings = torch.cat(
            [embeddings.sin(), embeddings.cos()],
            dim=1,
        )

        return embeddings


class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        self.time_projection = nn.Linear(
            time_dim,
            out_channels,
        )

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        projected_time = self.time_projection(time_embedding)
        projected_time = projected_time[:, :, None, None]

        h = h + projected_time

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + self.residual(x)


class SimpleUNet(nn.Module):
    """Small U-Net used for MNIST DDPM noise prediction."""

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 128,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.encoder1 = ResBlock(
            base_channels,
            base_channels,
            time_dim,
        )

        self.encoder2 = ResBlock(
            base_channels,
            base_channels * 2,
            time_dim,
        )

        self.bottleneck = ResBlock(
            base_channels * 2,
            base_channels * 4,
            time_dim,
        )

        self.decoder2 = ResBlock(
            base_channels * 4 + base_channels * 2,
            base_channels * 2,
            time_dim,
        )

        self.decoder1 = ResBlock(
            base_channels * 2 + base_channels,
            base_channels,
            time_dim,
        )

        self.downsample = nn.MaxPool2d(2)

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest",
        )

        self.output_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)

        x = self.input_conv(x)

        encoder1 = self.encoder1(x, time_embedding)

        encoder2 = self.encoder2(
            self.downsample(encoder1),
            time_embedding,
        )

        bottleneck = self.bottleneck(
            self.downsample(encoder2),
            time_embedding,
        )

        decoder2_input = torch.cat(
            [self.upsample(bottleneck), encoder2],
            dim=1,
        )

        decoder2 = self.decoder2(
            decoder2_input,
            time_embedding,
        )

        decoder1_input = torch.cat(
            [self.upsample(decoder2), encoder1],
            dim=1,
        )

        decoder1 = self.decoder1(
            decoder1_input,
            time_embedding,
        )

        return self.output_conv(decoder1)