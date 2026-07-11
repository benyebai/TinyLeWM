"""Run a short varied-batch training test and check basic stability."""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.smb_dataset import SMBSubTrajectoryDataset
from models.jepa import Jepa
from models.sigreg import SigReg
from scripts.train_jepa import get_lr_scheduler, train_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=Path("data/smb.h5"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("steps must be at least 2")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SMBSubTrajectoryDataset(args.h5)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    jepa = Jepa().to(device)
    sigreg = SigReg().to(device)
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = get_lr_scheduler(
        optimizer,
        total_steps=args.steps,
        warmup_steps=max(1, args.steps // 10),
    )

    pred_history: list[float] = []
    sigreg_history: list[float] = []
    step = 0

    print(
        f"Running {args.steps} smoke-test steps with batch size "
        f"{args.batch_size} on {device.type}"
    )

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            jepa.train()
            pixels = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            total_loss, pred_loss, sigreg_loss = train_step(
                jepa,
                sigreg,
                pixels,
                actions,
            )

            metrics = (total_loss, pred_loss, sigreg_loss)
            if not all(math.isfinite(metric.item()) for metric in metrics):
                raise SystemExit(f"FAIL: non-finite loss at step {step + 1}")

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                jepa.parameters(), max_norm=1.0
            )
            if not math.isfinite(grad_norm.item()):
                raise SystemExit(f"FAIL: non-finite gradient at step {step + 1}")

            optimizer.step()
            scheduler.step()
            step += 1

            pred_history.append(pred_loss.item())
            sigreg_history.append(sigreg_loss.item())

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(
                    f"step={step:4d} "
                    f"loss={total_loss.item():.4f} "
                    f"pred={pred_loss.item():.4f} "
                    f"sigreg={sigreg_loss.item():.4f} "
                    f"grad_norm={grad_norm.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

    dataset.close()

    window = min(20, len(pred_history) // 2)
    first_pred = sum(pred_history[:window]) / window
    last_pred = sum(pred_history[-window:]) / window
    last_sigreg = sum(sigreg_history[-window:]) / window

    print(f"First-{window} mean prediction loss: {first_pred:.6f}")
    print(f"Last-{window} mean prediction loss:  {last_pred:.6f}")
    print(f"Last-{window} mean SIGReg loss:      {last_sigreg:.6f}")

    if last_pred >= first_pred:
        raise SystemExit("FAIL: prediction loss did not improve")
    if last_sigreg >= 10.0:
        raise SystemExit("FAIL: SIGReg indicates likely representation collapse")

    print("PASS: losses stayed finite, prediction improved, and SIGReg stayed bounded")


if __name__ == "__main__":
    main()
