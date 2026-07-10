import torch
import torch.nn as nn

from models.action_encoder import ActionEmbedder
from models.encoder import Encoder
from models.predictor import ARPredictor


class Jepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.action_encoder = ActionEmbedder()
        self.predictor = ARPredictor()

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        embedded_frames = self.encoder(x)
        embedded_actions = self.action_encoder(x)
        predicted_frames = self.predictor(embedded_frames, embedded_actions)
        return predicted_frames
