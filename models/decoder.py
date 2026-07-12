"""Post-hoc convolutional decoder for visualizing JEPA latent states."""

import torch
import torch.nn as nn


def upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.GELU(),
    )


class PixelDecoder(nn.Module):
    """Turn one 192-dimensional JEPA state into a 224x224 RGB image."""

    def __init__(self, latent_dim: int = 192) -> None:
        super().__init__()
        self.to_feature_map = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.GELU(),
        )
        self.upsampler = nn.Sequential(
            upsample_block(256, 256),  # 7 -> 14
            upsample_block(256, 128),  # 14 -> 28
            upsample_block(128, 64),   # 28 -> 56
            upsample_block(64, 32),    # 56 -> 112
            upsample_block(32, 16),    # 112 -> 224
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.to_feature_map(latent)
        features = features.reshape(latent.size(0), 256, 7, 7)
        return self.upsampler(features)
