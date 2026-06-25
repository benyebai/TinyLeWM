import torch
import torch.nn as nn


# our embedding from our video encoder is [128, 4, 192]
class ARPredictor(nn.Module):  # autoregressive predictor
    def __init__(self, num_frame=4, input_dim=192):
        super().__init__()

        # this is for adding our position embedding, so each 128 batches has a group of 4
        # we want these 4 to learn the ordering, so we will create a 4 different 192 vectors
        self.positional_emb = nn.Parameter(torch.randn(1, num_frame, input_dim))

    def forward(self, x):
        T = x.size(1)  # get the T from the input
        # just means get everything but for the middle dim get the first T
        x = x + self.positional_emb[:, :T]

        return x
