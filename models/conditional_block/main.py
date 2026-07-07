import torch
import torch.nn as nn

from conditional_block.attention import Attention

class ConditionalBlock(nn.Module):
    def __init__(self, init_dim):
        super().__init__()
        # the actual tranformer blocks now (and just quickly we always basically do muilti-headed)
        # think a single attention KQV, then just one guys opinion kinda sux, we want muitiple people
        # learning their part and then combining them!
        self.attention = Attention(init_dim)

        # this is the mlp feed forward (supposedly super standard)
        # now u need activation sandwiched between 2 linears duh
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(init_dim),  # lets standarize our inputs first
            nn.Linear(init_dim, 2048),
            nn.GELU(),
            # this first dropout is to make sure the actual internal scratch board isnt dependent on something
            nn.Dropout(p=0.1),
            nn.Linear(2048, init_dim),
            # later this result will be the residual, we also dont want later attention blocks to be dependent
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor):
