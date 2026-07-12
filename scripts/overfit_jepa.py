"""Overfit a fixed set of SMB samples before attempting a larger training run."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from datasets.smb_dataset import SMBSubTrajectoryDataset
from models.jepa import Jepa
from models.sigreg import SigReg
from scripts.train_jepa import get_lr_scheduler, optimization_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=Path("data/smb.h5"))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--target-pred", type=float, default=0.01)
    args = parser.parse_args()

    if args.samples < 2:
        raise ValueError("The overfit test needs at least two samples")
    if args.steps < 2:
        raise ValueError("steps must be at least 2")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = (
        "bf16"
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else "float32"
    )

    dataset = SMBSubTrajectoryDataset(args.h5)
    if args.samples > len(dataset):
        raise ValueError(f"Requested {args.samples} samples, but dataset has {len(dataset)}")

    fixed_samples = Subset(dataset, range(args.samples))
    batch = next(
        iter(DataLoader(fixed_samples, batch_size=args.samples, shuffle=False))
    )
    pixels = batch["frames"].to(device)
    actions = batch["actions"].to(device)

    jepa = Jepa().to(device)
    sigreg = SigReg().to(device)
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = get_lr_scheduler(
        optimizer,
        total_steps=args.steps,
        warmup_steps=max(1, args.steps // 10),
    )

    initial_pred = None
    best_pred = float("inf")

    print(
        f"Overfitting {args.samples} fixed samples for {args.steps} steps "
        f"on {device.type} using {precision}"
    )

    for step in range(1, args.steps + 1):
        total_loss, pred_loss, sigreg_loss, grad_norm = optimization_step(
            jepa,
            sigreg,
            optimizer,
            scheduler,
            pixels,
            actions,
        )

        pred_value = pred_loss.item()
        if initial_pred is None:
            initial_pred = pred_value
        best_pred = min(best_pred, pred_value)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step={step:4d} "
                f"loss={total_loss.item():.4f} "
                f"pred={pred_value:.4f} "
                f"sigreg={sigreg_loss.item():.4f} "
                f"grad_norm={grad_norm.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    dataset.close()

    assert initial_pred is not None
    reduction = 100.0 * (1.0 - best_pred / initial_pred)
    print(
        f"Best prediction loss: {best_pred:.6f} "
        f"({reduction:.1f}% below the first step)"
    )

    if best_pred <= args.target_pred:
        print(f"PASS: prediction loss reached the {args.target_pred:g} target")
    else:
        raise SystemExit(
            f"FAIL: prediction loss did not reach the {args.target_pred:g} target"
        )


if __name__ == "__main__":
    main()
