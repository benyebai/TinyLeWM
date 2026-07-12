"""Evaluate a trained JEPA on held-out SMB episodes."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_batches < 1:
        raise ValueError("max-batches must be positive")

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(args.h5, "r") as h5:
        num_episodes = len(h5["episodes"])
    _, val_episode_indices = make_episode_split(
        num_episodes,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )

    dataset = SMBSubTrajectoryDataset(
        args.h5,
        episode_indices=val_episode_indices,
    )
    validation_generator = torch.Generator().manual_seed(0)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # SIGReg measures the distribution across a batch. Mix windows from
        # different episodes instead of grouping nearly identical neighbors.
        shuffle=True,
        generator=validation_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    jepa = Jepa().to(device)
    sigreg = SigReg().to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    jepa.load_state_dict(checkpoint["jepa"])
    sigreg.load_state_dict(checkpoint["sigreg"])
    jepa.eval()
    sigreg.eval()

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    pred_sum = 0.0
    persistence_sum = 0.0
    shuffled_sum = 0.0
    sigreg_sum = 0.0
    samples_seen = 0
    batches_seen = 0

    print(
        f"Evaluating checkpoint step={checkpoint['global_step']} on "
        f"{len(val_episode_indices)} held-out episodes ({len(dataset)} windows)"
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= args.max_batches:
                break

            pixels = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            batch_size = pixels.size(0)

            with precision_context:
                emb, act_emb = jepa.encode(pixels, actions)
                predictions = jepa.predict(emb, act_emb)
                pred_loss = F.mse_loss(predictions[:, :-1], emb[:, 1:])

                persistence_loss = F.mse_loss(emb[:, :-1], emb[:, 1:])

                permutation = torch.randperm(batch_size, device=device)
                shuffled_actions = actions[permutation].flatten(start_dim=2)
                shuffled_act_emb = jepa.action_encoder(shuffled_actions)
                shuffled_predictions = jepa.predict(emb, shuffled_act_emb)
                shuffled_loss = F.mse_loss(
                    shuffled_predictions[:, :-1],
                    emb[:, 1:],
                )

                sigreg_loss = sigreg(emb.transpose(0, 1))

            pred_sum += pred_loss.item() * batch_size
            persistence_sum += persistence_loss.item() * batch_size
            shuffled_sum += shuffled_loss.item() * batch_size
            sigreg_sum += sigreg_loss.item() * batch_size
            samples_seen += batch_size
            batches_seen += 1

    dataset.close()

    pred_mean = pred_sum / samples_seen
    persistence_mean = persistence_sum / samples_seen
    shuffled_mean = shuffled_sum / samples_seen
    sigreg_mean = sigreg_sum / samples_seen
    persistence_improvement = 100.0 * (1.0 - pred_mean / persistence_mean)
    shuffled_increase = 100.0 * (shuffled_mean / pred_mean - 1.0)

    print(f"batches={batches_seen} samples={samples_seen}")
    print(f"trained prediction loss:  {pred_mean:.6f}")
    print(f"persistence baseline loss: {persistence_mean:.6f}")
    print(f"shuffled-action loss:      {shuffled_mean:.6f}")
    print(f"validation SIGReg loss:    {sigreg_mean:.6f}")
    print(f"improvement vs persistence: {persistence_improvement:.1f}%")
    print(f"loss increase when actions are shuffled: {shuffled_increase:.1f}%")


if __name__ == "__main__":
    main()
