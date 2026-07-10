# will use all the stuff we have in models and our dataset to run the training loop
import torch
import torch.nn.functional as F

from models.jepa import Jepa
from models.sigreg import SigReg


def train_step(jepa: Jepa, sigreg: SigReg, pixels, actions, lmbda=0.1):
    # pixels: [B, T, C, H, W]
    # actions: [B, T, 30]

    emb, emb_act = jepa.encode(pixels, actions)  # [B, T, D]

    next_emb = jepa.predict(emb, emb_act)

    # Pred loss -- MSE, matching time time steps
    # t0 |t1 t2 t3| vs |t1 t2 t3| t4
    pred_loss = F.mse_loss(next_emb[:, :-1], emb[:, 1:])
    # needs to be in [T, B, D]
    sigreg_loss = sigreg(emb.transpose(0, 1))
    total_loss = pred_loss + lmbda * sigreg_loss

    return total_loss, pred_loss, sigreg_loss
