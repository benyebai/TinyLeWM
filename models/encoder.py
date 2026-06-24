import torch.nn as nn
from transformers import ViTConfig, ViTModel


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel(
            ViTConfig(
                image_size=224,
                patch_size=14,
                hidden_size=192,
                num_hidden_layers=12,
                num_attention_heads=3,
            )
        )

        self.projector = nn.Sequential(nn.Linear(192, 192), nn.BatchNorm1d(192))

    def forward(self, x):
        cls_token = self.vit(x).last_hidden_state[:, 0]
        return self.projector(cls_token)
