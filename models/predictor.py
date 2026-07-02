import torch
import torch.nn as nn


# our embedding from our video encoder is [128, 4, 192]
class ARPredictor(nn.Module):  # autoregressive predictor
    def __init__(self, num_frame=4, input_dim=192):
        super().__init__()

        # this is for adding our position embedding, so each 128 batches has a group of 4
        # these 4 are tags we learn, best way to represent these 4 ordered tags,
        # so we will create a 4 different 192 vectors and then use it for all batches
        self.positional_emb = nn.Parameter(torch.randn(1, num_frame, input_dim))
        self.dropout = nn.Dropout(p=0.1)

        # this should just be a simple projector to our 192 space if its not 192
        # self.input_proj
        # self.cond_proj

        # the actual tranformer blocks now (and just quickly we always basically do muilti-headed)
        # think a single attention KQV, then just one guys opinion kinda sux, we want muitiple people
        # learning their part and then combining them!

        self.attention = nn.Sequential(
            nn.LayerNorm(input_dim),  # lets standarize our inputs first
        )

        # this is the mlp feed forward (supposedly super standard)
        # now u need activation sandwiched between 2 linears duh
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(input_dim),  # lets standarize our inputs first
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            # this first dropout is to make sure the actual internal scratch board isnt dependent on something
            nn.Dropout(p=0.1),
            nn.Linear(2048, input_dim),
            # later this result will be the residual, we also dont want later attention blocks to be dependent
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        T = x.size(1)  # get the T from the input
        # just means get everything but for the middle dim get the first T
        x = x + self.positional_emb[:, :T]
        # this is the dropout, we just zero out some features in our embedding tensors
        # it might learn based on just our data, we need it to learn more genreally
        # get to an answer from multiple routes
        x = self.dropout(x)
        return x
