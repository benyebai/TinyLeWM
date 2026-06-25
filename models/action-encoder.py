import torch
import torch.nn as nn


# the action embedder, should take our raw actions [B, T, 30] (since 6 buttons per frame, and each block is 5 frames)
# then turn it into [B, T, 192] (because we want the actions to be embedded into the same space as our frame embedding (also 192))
class ActionEmbedder(nn.Module):
    def __init__(self, input_dim=30, smoothed_dim=64, emb_dim=192, mlp_scale=4):
        super().__init__()

        # we first need a conv1d
        # then we need a 2-layer MLP
        #
        # follows the actual leworldmodel implementation,
        # dosent actually need to be this complicated but just following standards
        #
        # in my understanding the embedder is just learning whats the most useful embedding forms so that
        # the predictor can make better judgements based on our actions
        self.conv = nn.Conv1d(
            in_channels=input_dim, out_channels=smoothed_dim, kernel_size=1, stride=1
        )
        self.mlp = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    # takes in [B, T, 30]
    def forward(self, x):
        x = x.float()
        x = x.transpose(1, 2)  # from [B, T, 30] -> [B, 30, T]
        x = self.conv(x)
        x = x.transpose(1, 2)  # revert back
        return self.mlp(x)
