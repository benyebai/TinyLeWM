"""Validate the deterministic episode-level train/validation split."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from datasets.smb_dataset import SMBSubTrajectoryDataset
from datasets.splits import make_episode_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=Path("data/smb-full.h5"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as h5:
        episodes = h5["episodes"][:]

    train_indices, val_indices = make_episode_split(
        len(episodes),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_again, val_again = make_episode_split(
        len(episodes),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    assert np.array_equal(train_indices, train_again)
    assert np.array_equal(val_indices, val_again)
    assert np.intersect1d(train_indices, val_indices).size == 0
    assert np.union1d(train_indices, val_indices).size == len(episodes)

    train_dataset = SMBSubTrajectoryDataset(
        args.h5,
        episode_indices=train_indices,
    )
    val_dataset = SMBSubTrajectoryDataset(
        args.h5,
        episode_indices=val_indices,
    )

    episode_starts = episodes[:, 0]
    train_owners = np.searchsorted(episode_starts, train_dataset.starts, side="right") - 1
    val_owners = np.searchsorted(episode_starts, val_dataset.starts, side="right") - 1

    assert np.isin(train_owners, train_indices).all()
    assert np.isin(val_owners, val_indices).all()
    assert not np.isin(train_owners, val_indices).any()
    assert not np.isin(val_owners, train_indices).any()

    print("Episode split ok")
    print(f"  seed: {args.seed}")
    print(f"  train episodes: {len(train_indices)}")
    print(f"  validation episodes: {len(val_indices)}")
    print(f"  train windows: {len(train_dataset)}")
    print(f"  validation windows: {len(val_dataset)}")

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    main()
