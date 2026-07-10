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
        self.t = torch.linspace(0, 3, knots)
        self.dt = 3 / (knots - 1)
        self.window = torch.exp(-self.t**2 / 2) # the phi
        self.weights = torch.full((17,), self.dt * 2)
        self.weights[0], self.weights[-1] = self.dt, self.dt


        self.register_buffer("t", self.t)
        self.register_buffer("window", self.window)
        self.register_buffer("weights", self.weights)


    def forward(self, emb):
        # for each arrow through the cloud of embeddings
        #   do the dot product for each embedding
        #       sum [B, T, 192], for each of the embeddings cos(t * x) / BxT for the average
