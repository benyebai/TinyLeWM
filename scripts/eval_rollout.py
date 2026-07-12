"""Evaluate recursive five-step JEPA rollouts on held-out SMB episodes."""

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


def rollout_five_steps(
    jepa: Jepa,
    initial_states: torch.Tensor,
    action_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Use three trained context positions, then imagine five more."""
    history = initial_states
    for future_index in range(4, 9):
        action_history = action_embeddings[:, future_index - 3 : future_index]
        predicted_sequence = jepa.predict(history, action_history)
        next_state = predicted_sequence[:, -1:]
        history = torch.cat((history[:, 1:], next_state), dim=1)
    return history[:, -1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

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
        num_frames=9,
        episode_indices=val_episode_indices,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    jepa = Jepa().to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    jepa.load_state_dict(checkpoint["jepa"])
    jepa.eval()

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    rollout_sum = 0.0
    baseline_sum = 0.0
    shuffled_sum = 0.0
    samples_seen = 0
    batches_seen = 0

    print(
        f"Evaluating five-step rollout from checkpoint step="
        f"{checkpoint['global_step']} on {len(val_episode_indices)} held-out episodes"
    )

    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= args.max_batches:
                break

            pixels = batch["frames"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            batch_size = pixels.size(0)

            with precision_context:
                embeddings, action_embeddings = jepa.encode(pixels, actions)
                prediction = rollout_five_steps(
                    jepa,
                    embeddings[:, 1:4],
                    action_embeddings,
                )
                target = embeddings[:, 8]
                current_state = embeddings[:, 3]

                permutation = torch.randperm(batch_size, device=device)
                shuffled_prediction = rollout_five_steps(
                    jepa,
                    embeddings[:, 1:4],
                    action_embeddings[permutation],
                )

                rollout_loss = F.mse_loss(prediction, target)
                baseline_loss = F.mse_loss(current_state, target)
                shuffled_loss = F.mse_loss(shuffled_prediction, target)

            rollout_sum += rollout_loss.item() * batch_size
            baseline_sum += baseline_loss.item() * batch_size
            shuffled_sum += shuffled_loss.item() * batch_size
            samples_seen += batch_size
            batches_seen += 1

    dataset.close()

    rollout_mean = rollout_sum / samples_seen
    baseline_mean = baseline_sum / samples_seen
    shuffled_mean = shuffled_sum / samples_seen
    improvement = 100.0 * (1.0 - rollout_mean / baseline_mean)
    shuffled_increase = 100.0 * (shuffled_mean / rollout_mean - 1.0)

    print(f"batches={batches_seen} samples={samples_seen}")
    print(f"five-step rollout loss:    {rollout_mean:.6f}")
    print(f"no-change baseline loss:   {baseline_mean:.6f}")
    print(f"shuffled-action loss:      {shuffled_mean:.6f}")
    print(f"improvement vs no-change:  {improvement:.1f}%")
    print(f"loss increase with shuffled actions: {shuffled_increase:.1f}%")


if __name__ == "__main__":
    main()
