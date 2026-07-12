# will use all the stuff we have in models and our dataset to run the training loop
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.smb_dataset import SMBSubTrajectoryDataset
from datasets.splits import make_episode_split
from models.jepa import Jepa
from models.sigreg import SigReg


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON record as a single line, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def save_checkpoint(
    path: Path,
    jepa: torch.nn.Module,
    sigreg: SigReg,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    config: dict,
) -> None:
    """Atomically save everything needed to continue training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "jepa": jepa.state_dict(),
            "sigreg": sigreg.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    jepa: torch.nn.Module,
    sigreg: SigReg,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device | str,
) -> dict:
    """Restore training state from a trusted checkpoint file."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    jepa.load_state_dict(checkpoint["jepa"])
    sigreg.load_state_dict(checkpoint["sigreg"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


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


def optimization_step(
    jepa: Jepa,
    sigreg: SigReg,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    max_grad_norm: float = 1.0,
    lmbda: float = 0.1,
    use_bf16: bool = True,
):
    """Run one complete model update and return detached metrics."""
    jepa.train()
    optimizer.zero_grad(set_to_none=True)

    bf16_enabled = (
        use_bf16
        and pixels.device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16_enabled
        else nullcontext()
    )

    with precision_context:
        total_loss, pred_loss, sigreg_loss = train_step(
            jepa,
            sigreg,
            pixels,
            actions,
            lmbda=lmbda,
        )

    losses = (total_loss, pred_loss, sigreg_loss)
    if not all(torch.isfinite(loss).item() for loss in losses):
        raise FloatingPointError("Training produced a non-finite loss")

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        jepa.parameters(),
        max_norm=max_grad_norm,
    )
    if not torch.isfinite(grad_norm).item():
        raise FloatingPointError("Training produced a non-finite gradient norm")

    optimizer.step()
    scheduler.step()

    return (
        total_loss.detach(),
        pred_loss.detach(),
        sigreg_loss.detach(),
        grad_norm.detach(),
    )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=Path("data/smb.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    h5_path = args.h5
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    warmup_steps = args.warmup_steps
    max_grad_norm = args.max_grad_norm
    log_every = args.log_every
    checkpoint_every = args.checkpoint_every
    checkpoint_dir = args.output_dir
    metrics_path = checkpoint_dir / "metrics.jsonl"

    with h5py.File(h5_path, "r") as h5:
        num_episodes = len(h5["episodes"])
    train_episode_indices, val_episode_indices = make_episode_split(
        num_episodes,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )

    dataset = SMBSubTrajectoryDataset(
        h5_path,
        episode_indices=train_episode_indices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    steps_from_epochs = num_epochs * len(dataloader)
    total_steps = (
        min(args.max_steps, steps_from_epochs)
        if args.max_steps is not None
        else steps_from_epochs
    )
    if total_steps <= 0:
        raise ValueError("Training requires at least one step")

    training_config = {
        "h5_path": str(h5_path),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = (
        "bf16"
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else "float32"
    )
    print(f"device={device.type} precision={precision}")
    jepa = Jepa().to(device)
    sigreg = SigReg().to(device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # we are using the adamw algorithm
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Think about whether warmup_steps is valid for the small sample HDF5 file.
    warmup_steps_real = min(warmup_steps, total_steps//10)
    scheduler = get_lr_scheduler(optimizer=optimizer, total_steps=total_steps, warmup_steps=warmup_steps_real)

    global_step = 0
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            jepa,
            sigreg,
            optimizer,
            scheduler,
            device,
        )
        saved_config = checkpoint["config"]
        for key in ("batch_size", "learning_rate", "weight_decay", "total_steps"):
            if saved_config.get(key) != training_config[key]:
                raise ValueError(f"Resume configuration mismatch for {key}")
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"]
        print(f"resumed={args.resume} step={global_step} epoch={start_epoch + 1}")

    print(
        f"train_episodes={len(train_episode_indices)} "
        f"validation_episodes={len(val_episode_indices)} "
        f"train_windows={len(dataset)} total_steps={total_steps}"
    )

    last_epoch = start_epoch
    previous_step_finished = time.monotonic()
    for epoch in range(start_epoch, num_epochs):
        last_epoch = epoch
        jepa.train()

        for batch in dataloader:
            if global_step >= total_steps:
                break

            batch_ready = time.monotonic()
            data_seconds = batch_ready - previous_step_finished
            pixels = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)

            update_started = time.monotonic()
            total_loss, pred_loss, sigreg_loss, grad_norm = optimization_step(
                jepa,
                sigreg,
                optimizer,
                scheduler,
                pixels,
                actions,
                max_grad_norm=max_grad_norm,
            )
            update_seconds = time.monotonic() - update_started
            previous_step_finished = time.monotonic()

            global_step += 1

            if global_step == 1 or global_step % log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                metrics = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": total_loss.item(),
                    "pred_loss": pred_loss.item(),
                    "sigreg_loss": sigreg_loss.item(),
                    "grad_norm": grad_norm.item(),
                    "learning_rate": current_lr,
                    "data_seconds": data_seconds,
                    "update_seconds": update_seconds,
                }
                append_jsonl(metrics_path, metrics)
                print(
                    f"epoch={epoch + 1}/{num_epochs} "
                    f"step={global_step}/{total_steps} "
                    f"loss={total_loss.item():.4f} "
                    f"pred={pred_loss.item():.4f} "
                    f"sigreg={sigreg_loss.item():.4f} "
                    f"grad_norm={grad_norm.item():.4f} "
                    f"lr={current_lr:.2e} "
                    f"data_s={data_seconds:.3f} "
                    f"update_s={update_seconds:.3f}"
                )

            if global_step % checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"jepa_step_{global_step}.pt",
                    jepa,
                    sigreg,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    training_config,
                )

        if global_step >= total_steps:
            break

    # Always save the final state, even for a short test run.
    save_checkpoint(
        checkpoint_dir / "jepa_final.pt",
        jepa,
        sigreg,
        optimizer,
        scheduler,
        last_epoch,
        global_step,
        training_config,
    )


if __name__ == "__main__":
    main()
