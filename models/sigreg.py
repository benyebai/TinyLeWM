import torch
import torch.nn as nn

# Every world model loss function is trying to avoid representation collapse and lop sidedness
#
# SIGReg now what is it? In my understanding lets say u have an array of numbers, to figure out if its gaussian
# if u only calculate the mean and the sd, and look for if its close to 0 or 1, u can get tricked [-1, -1, 1, 1] (not gaussian but satisfies the requirements)
# so what we do instead is fit it using the functions cos(t * x) / sin(t * x), then the average is exp(-t**2 / 2) if its gaussian
# and u want to do that accross all directions
#
# For terminology: characteristic function - thats the cos average or the sin average
#                  epps-pulley test - comparing it to exp(-t**2 / 2)
#
# and why we need to use cos/sin its because sin catches lenaning left or right, cos catches are they spread/clumped properly


class SigReg(nn.Module):
    # num_arrows: how many vectors we are going to shoot into the cloud of embeddigns
    # knots: how many t's we r going to check per direction
    # t: the actual values of the t's
    def __init__(self, knots=17, num_arrows=1024):
        super().__init__()
        self.num_arrows = num_arrows
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        window = torch.exp(-(t**2) / 2)  # the phi
        weights = torch.full((knots,), dt * 2)
        weights[0], weights[-1] = dt, dt
        weights = (
            weights * window
        )  # this calculates the importance of that t in our final weight
        # because u are adding all the t's together, but some of them are more usless than others

        self.register_buffer("t", t)
        self.register_buffer("window", window)
        self.register_buffer("weights", weights)

    # NOTE: we are checking if each time steps is gaussian, so group t1 together, then t2, etc
    def forward(self, emb):  # emb will come in as [T, B, D]

        # for each arrow through the cloud of embeddings
        # A: [D, 1024]
        A = torch.randn(emb.size(-1), self.num_arrows, device=emb.device)
        A = A / A.norm(p=2, dim=0)

        #   do the dot product for each embedding
        shadows = emb @ A
        #       sum [B, T, 192], for each of the embeddings cos(t * x) / BxT for the average (and all these steps for sin)
        x_t = shadows.unsqueeze(-1) * self.t
        cos_avg = x_t.cos().mean(dim=1)
        sin_avg = x_t.sin().mean(dim=1)
        #       compare to target and get error score
        err = (cos_avg - self.window).square() + sin_avg.square()
        # then turn it into one number (get weighted sum, then average it)
        statistic = (err @ self.weights) * emb.size(1)  # (T, num_arrows)
        return statistic.mean()
