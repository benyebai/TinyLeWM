# will use all the stuff we have in models and our dataset to run the training loop
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.smb_dataset import SMBSubTrajectoryDataset
from models.jepa import Jepa
from models.sigreg import SigReg


def train_step(jepa: Jepa, sigreg: SigReg, pixels, actions, lmbda=0.1):
    # pixels: [B, T, C, H, W]
    # actions: [B, T, 5, 6]

    emb, emb_act = jepa.encode(pixels, actions)  # [B, T, D]

    next_emb = jepa.predict(emb, emb_act)

    # t0 |t1 t2 t3| vs |t1 t2 t3| t4
    pred_loss = F.mse_loss(next_emb[:, :-1], emb[:, 1:])
    # needs to be in [T, B, D]
    sigreg_loss = sigreg(emb.transpose(0, 1))
    total_loss = pred_loss + lmbda * sigreg_loss

    return total_loss, pred_loss, sigreg_loss


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
):
    # so i guess we linearly warm up to the base LR, then cosine-decay toward zero
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps")

    def lr_multiplier(step: int) -> float:
        # LambdaLR multiplies the optimizer's original learning rate by this value.
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)

        decay_steps = total_steps - warmup_steps
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )


def main():
    # We will eventually move these into a config or command-line arguments.
    h5_path = "data/smb.h5"
    batch_size = 128
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 0.05
    warmup_steps = 1000
    max_grad_norm = 1.0
    log_every = 10
    checkpoint_every = 5000
    checkpoint_dir = Path("checkpoints")

    dataset = SMBSubTrajectoryDataset(h5_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jepa = Jepa().to(device)
    sigreg = SigReg().to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # we are using the adamw algorithm
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = num_epochs * len(dataloader)

    # Think about whether warmup_steps is valid for the small sample HDF5 file.
    warmup_steps_real = min(warmup_steps, total_steps//10)
    scheduler = get_lr_scheduler(optimizer=optimizer, total_steps=total_steps, warmup_steps=warmup_steps_real)

    global_step = 0
    for epoch in range(num_epochs):
        jepa.train()

        for batch in dataloader:
            pixels = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            # clear the gradients left over from the previous step.
            optimizer.zero_grad(set_to_none=True)
            # get the 3 losses
            total_loss, pred_loss, sigreg_loss = train_step(
                jepa,
                sigreg,
                pixels,
                actions,
            )
            # backpropagate the total loss. calculates and stores gradients
            total_loss.backward()
            # basically jepa's gradient can get too big the steps so we want them less than 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                jepa.parameters(),
                max_norm=max_grad_norm,
            )
            # Update the model parameters.
            optimizer.step()

            # Update the learning rate for the next iteration.
            scheduler.step()

            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch + 1}/{num_epochs} "
                    f"step={global_step}/{total_steps} "
                    f"loss={total_loss.item():.4f} "
                    f"pred={pred_loss.item():.4f} "
                    f"sigreg={sigreg_loss.item():.4f} "
                    f"grad_norm={grad_norm.item():.4f} "
                    f"lr={current_lr:.2e}"
                )

            if global_step % checkpoint_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "jepa": jepa.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    checkpoint_dir / f"jepa_step_{global_step}.pt",
                )

    # Always save the final state, even for a short test run.
    torch.save(
        {
            "epoch": num_epochs - 1,
            "global_step": global_step,
            "jepa": jepa.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "jepa_final.pt",
    )


if __name__ == "__main__":
    main()
