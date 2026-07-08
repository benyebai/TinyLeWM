import torch
import torch.nn as nn

from models.conditional_block.main import ConditionalBlock


# our embedding from our video encoder is [128, 4, 192]
class ARPredictor(nn.Module):  # autoregressive predictor
    def __init__(self, num_frame=4, input_dim=192):
        super().__init__()

        # this is for adding our position embedding, so each 128 batches has a group of 4
        # these 4 are tags we learn, best way to represent these 4 ordered tags,
        # so we will create a 4 different 192 vectors and then use it for all batches
        self.positional_emb = nn.Parameter(torch.randn(1, num_frame, input_dim))
        self.dropout = nn.Dropout(p=0.1)

        # pytorch note: sequential only pipes 1 singular tensor through, so modulelist isntead
        self.blocks = nn.ModuleList([ConditionalBlock(input_dim) for _ in range(6)])

    def forward(self, x, c):
        T = x.size(1)  # get the T from the input
        # just means get everything but for the middle dim get the first T
        x = x + self.positional_emb[:, :T]
        # this is the dropout, we just zero out some features in our embedding tensors
        # it might learn based on just our data, we need it to learn more genreally
        # get to an answer from multiple routes
        x = self.dropout(x)

        for b in self.blocks:
            x = b(x, c)

        return x
