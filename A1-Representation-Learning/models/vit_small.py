import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x shape: [batch_size, 3, 32, 32]

        x = self.proj(x)
        # shape: [batch_size, embed_dim, 8, 8]

        x = x.flatten(2)
        # shape: [batch_size, embed_dim, num_patches]

        x = x.transpose(1, 2)
        # shape: [batch_size, num_patches, embed_dim]

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=512, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = x + residual

        # MLP block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class ViTSmall(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_dim=512,
        dropout=0.1
    ):
        super(ViTSmall, self).__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.patch_embed(x)
        # shape: [batch_size, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # shape: [batch_size, 1, embed_dim]

        x = torch.cat((cls_tokens, x), dim=1)
        # shape: [batch_size, num_patches + 1, embed_dim]

        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)

        x = self.norm(x)

        cls_output = x[:, 0]
        # take class token output

        x = self.head(cls_output)

        return x