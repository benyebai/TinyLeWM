import torch
import torch.nn as nn
from conditional_block.attention import Attention


# From my understanding this whole conditional block is we want to introduce 3 different transformations ontop of our predictor
# To understand it intuitively:
#
# This block will be ran 6 times, each time the next embedding will be steered towards the predicted latent embedding
# 1) attention block gathers a summary and context of whats happening on screen
# 2) the mlp then tries to turn that into a concept an understanding of the world and predicts a piece of the next latent embedding
# 3) rinse and repeate 6 times and the whole picture will be pieced together
#
# Now what does the conditional do:
# scale: which can turn specific features high or low, so if its an up input, turn the vertical motion features way up
# shift: its just adds constants onto the features
# gate: now after attention or the FFN, perhaps u think the change is not important, x = x + gate * change
# so now our pieces can be modified based on the action as well
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

        # self.thing = get 6 different weights

    # x is the embedded frames, c is the associated emebedded actions
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # a b c d e f = self.thing(c)

        x = self.attention(x)
        x = x + c * (a * (x + b))
        x = self.feed_forward(x)
        x = x + f * (d * (x + e))
        return x
