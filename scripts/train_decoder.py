"""Train the post-hoc pixel decoder while keeping the JEPA frozen."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from datasets.smb_dataset import SMBSubTrajectoryDataset
from datasets.splits import make_episode_split
from models.decoder import PixelDecoder
from models.jepa import Jepa


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(
            (step - warmup_steps) / max(1, total_steps - warmup_steps),
            1.0,
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    hard_pixel_weight: float,
    edge_weight: float,
    hard_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel_mse = F.mse_loss(reconstruction, target)

    per_pixel_error = (reconstruction - target).square().mean(dim=1)
    flat_error = per_pixel_error.flatten(start_dim=1)
    hard_count = max(1, int(flat_error.size(1) * hard_fraction))
    hard_pixel_loss = flat_error.topk(hard_count, dim=1).values.mean()

    reconstruction_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    reconstruction_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    edge_loss = F.l1_loss(reconstruction_dx, target_dx) + F.l1_loss(
        reconstruction_dy,
        target_dy,
    )

    total = (
        pixel_mse
        + hard_pixel_weight * hard_pixel_loss
        + edge_weight * edge_loss
    )
    return total, pixel_mse, hard_pixel_loss, edge_loss


def save_decoder(
    path: Path,
    decoder: PixelDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def save_preview(
    targets: torch.Tensor,
    reconstructions: torch.Tensor,
    path: Path,
) -> None:
    frame_size = 224
    header_height = 28
    row_gap = 4
    rows = targets.size(0)
    canvas = Image.new(
        "RGB",
        (frame_size * 2, header_height + rows * frame_size + (rows - 1) * row_gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((82, 7), "REAL", fill="black")
    draw.text((294, 7), "RECONSTRUCTION", fill="black")

    def convert(tensor: torch.Tensor) -> Image.Image:
        pixels = tensor.detach().float().clamp(-1, 1)
        pixels = ((pixels + 1.0) * 127.5).byte()
        array = pixels.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")

    for row in range(rows):
        y = header_height + row * (frame_size + row_gap)
        canvas.paste(convert(targets[row]), (0, y))
        canvas.paste(convert(reconstructions[row]), (frame_size, y))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--hard-pixel-weight", type=float, default=0.0)
    parser.add_argument("--edge-weight", type=float, default=0.0)
    parser.add_argument("--hard-fraction", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_steps < 2:
        raise ValueError("max-steps must be at least 2")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with h5py.File(args.h5, "r") as h5:
        num_episodes = len(h5["episodes"])
    train_episode_indices, _ = make_episode_split(
        num_episodes,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )
    dataset = SMBSubTrajectoryDataset(
        args.h5,
        episode_indices=train_episode_indices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    jepa = Jepa().to(device)
    jepa_checkpoint = torch.load(
        args.jepa_checkpoint,
        map_location=device,
        weights_only=False,
    )
    jepa.load_state_dict(jepa_checkpoint["jepa"])
    jepa.eval()
    for parameter in jepa.parameters():
        parameter.requires_grad_(False)

    decoder = PixelDecoder().to(device)
    if args.decoder_checkpoint is not None:
        decoder_checkpoint = torch.load(
            args.decoder_checkpoint,
            map_location=device,
            weights_only=False,
        )
        decoder.load_state_dict(decoder_checkpoint["decoder"])
        print(
            f"Fine-tuning decoder from {args.decoder_checkpoint} "
            f"(saved step {decoder_checkpoint['step']})"
        )
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    warmup_steps = min(200, max(1, args.max_steps // 10))
    scheduler = cosine_scheduler(optimizer, args.max_steps, warmup_steps)

    fixed_batch = next(iter(dataloader)) if args.fixed_batch else None
    iterator = iter(dataloader)
    first_loss = None

    print(
        f"Training decoder for {args.max_steps} steps on {device.type} "
        f"using {'bf16' if use_bf16 else 'float32'} "
        f"({'fixed batch' if args.fixed_batch else 'varied batches'})"
    )

    for step in range(1, args.max_steps + 1):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)

        # One frame per window is enough; randomized windows cover the dataset.
        target = batch["frames"][:, 0].to(device, non_blocking=True)
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )

        with torch.no_grad(), precision_context:
            latent = jepa.encoder(target)

        optimizer.zero_grad(set_to_none=True)
        with precision_context:
            reconstruction = decoder(latent)
            loss, pixel_mse, hard_pixel_loss, edge_loss = reconstruction_loss(
                reconstruction,
                target,
                hard_pixel_weight=args.hard_pixel_weight,
                edge_weight=args.edge_weight,
                hard_fraction=args.hard_fraction,
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Decoder produced a non-finite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()
        if first_loss is None:
            first_loss = loss_value

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            print(
                f"step={step:5d}/{args.max_steps} "
                f"loss={loss_value:.6f} "
                f"pixel_mse={pixel_mse.item():.6f} "
                f"hard={hard_pixel_loss.item():.6f} "
                f"edge={edge_loss.item():.6f} "
                f"grad_norm={grad_norm.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if step % args.checkpoint_every == 0:
            save_decoder(
                args.output_dir / f"decoder_step_{step}.pt",
                decoder,
                optimizer,
                scheduler,
                step,
            )

    save_decoder(
        args.output_dir / "decoder_final.pt",
        decoder,
        optimizer,
        scheduler,
        args.max_steps,
    )
    reduction = 100.0 * (1.0 - loss_value / first_loss)
    print(f"training loss reduction: {reduction:.1f}%")

    if fixed_batch is not None:
        decoder.eval()
        target = fixed_batch["frames"][:, 0].to(device, non_blocking=True)
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with torch.inference_mode(), precision_context:
            latent = jepa.encoder(target)
            reconstruction = decoder(latent)
            shuffled_reconstruction = decoder(latent.roll(shifts=1, dims=0))
            clean_loss = F.mse_loss(reconstruction, target)
            shuffled_loss = F.mse_loss(shuffled_reconstruction, target)
        print(f"clean fixed-batch MSE:     {clean_loss.item():.6f}")
        print(f"shuffled-latent MSE:       {shuffled_loss.item():.6f}")
        if args.preview_output is not None:
            save_preview(target, reconstruction, args.preview_output)
            print(f"saved preview: {args.preview_output}")

    dataset.close()


if __name__ == "__main__":
    main()
