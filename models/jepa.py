import torch
import torch.nn as nn
from einops import rearrange

from models.action_encoder import ActionEmbedder
from models.encoder import Encoder
from models.predictor import ARPredictor


class Jepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.action_encoder = ActionEmbedder()
        self.predictor = ARPredictor()
        self.pred_proj = nn.Sequential(
            nn.Linear(192, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, 192)
        )

    def encode(self, pixels, actions):
        # pixels: [B, T, C, H, W]
        # actions: [B, T, 30]
        B = pixels.size(0)
        pixels = rearrange(pixels, "b t c h w -> (b t) c h w")
        emb = self.encoder(pixels)
        emb = rearrange(emb, "(b t) d -> b t d", b=B)
        act_emb = self.action_encoder(actions)
        return emb, act_emb

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor):
        preds = self.predictor(emb, act_emb)
        preds = rearrange(preds, "b t d -> (b t) d")
        preds = self.pred_proj(preds)
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds
