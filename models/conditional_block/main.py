import torch
import torch.nn as nn
from conditional_block.attention import Attention


# for the Adaptive LayerNorm
def modulate(t, shift, scale):
    return t * (1 + scale) + shift


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

        self.normalize = nn.LayerNorm(init_dim, elementwise_affine=False)

        # the actual tranformer blocks now (and just quickly we always basically do muilti-headed)
        # think a single attention KQV, then just one guys opinion kinda sux, we want muitiple people
        # learning their part and then combining them!
        self.attention = Attention(init_dim)

        # this is the mlp feed forward (supposedly super standard)
        # now u need activation sandwiched between 2 linears duh
        self.feed_forward = nn.Sequential(
            nn.Linear(init_dim, 2048),
            nn.GELU(),
            # this first dropout is to make sure the actual internal scratch board isnt dependent on something
            nn.Dropout(p=0.1),
            nn.Linear(2048, init_dim),
            # later this result will be the residual, we also dont want later attention blocks to be dependent
            nn.Dropout(p=0.1),
        )

        # creates the 6 modulations (but they are combined all into one)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(init_dim, 6 * init_dim)
        )

    # x is the embedded frames, c is the associated emebedded actions
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # [B, T, 192x6]
        modulations = self.adaLN_modulation(c)
        # each is [B, T, 192]
        a, b, c, d, e, f = modulations.chunk(6, -1)

        x = self.normalize(x)
        x = x + c * modulate(self.attention(x), a, b)
        x = self.normalize(x)
        x = x + f * modulate(self.feed_forward(x), d, e)

        return x
