"""
Basic validation for build_hdf5.py output and SMBSubTrajectoryDataset.

Usage:
    python -m datasets.validate_smb_dataset --h5 data/smb.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import DataLoader

from datasets.build_hdf5 import FRAME_C, FRAME_H, FRAME_W, META_DTYPE
from datasets.smb_dataset import SMBSubTrajectoryDataset


def validate_hdf5(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5:
        required = ("frames", "actions", "frame_metadata", "episodes")
        for name in required:
            if name not in h5:
                raise ValueError(f"Missing dataset: /{name}")

        frames = h5["frames"]
        actions = h5["actions"]
        metadata = h5["frame_metadata"]
        episodes = h5["episodes"]

        assert frames.ndim == 4
        assert frames.shape[1:] == (FRAME_H, FRAME_W, FRAME_C)
        assert frames.dtype == np.uint8

        assert actions.shape == (frames.shape[0], 6)
        assert actions.dtype == np.uint8

        assert metadata.shape == (frames.shape[0],)
        assert metadata.dtype == META_DTYPE

        assert episodes.ndim == 2
        assert episodes.shape[1] == 2
        assert int(episodes[:, 1].sum()) == frames.shape[0]

        print("HDF5 ok")
        print(f"  frames: {frames.shape} {frames.dtype}")
        print(f"  actions: {actions.shape} {actions.dtype}")
        print(f"  metadata: {metadata.shape}")
        print(f"  episodes: {episodes.shape[0]}")


def validate_dataset(h5_path: Path, batch_size: int) -> None:
    dataset = SMBSubTrajectoryDataset(h5_path)
    sample = dataset[0]

    assert sample["frames"].shape == (4, 3, 224, 224)
    assert sample["actions"].shape == (4, 5, 6)
    assert sample["frames"].dtype.is_floating_point
    assert sample["actions"].dtype.is_floating_point
    assert -1.0 <= float(sample["frames"].min()) <= 1.0
    assert -1.0 <= float(sample["frames"].max()) <= 1.0
    assert set(sample["actions"].unique().tolist()).issubset({0.0, 1.0})

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    assert batch["frames"].shape == (batch_size, 4, 3, 224, 224)
    assert batch["actions"].shape == (batch_size, 4, 5, 6)

    dataset.close()

    print("Dataset ok")
    print(f"  samples: {len(dataset)}")
    print(f"  sample frames: {tuple(sample['frames'].shape)}")
    print(f"  sample actions: {tuple(sample['actions'].shape)}")
    print(f"  batch frames: {tuple(batch['frames'].shape)}")
    print(f"  batch actions: {tuple(batch['actions'].shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=Path("data/smb.h5"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    validate_hdf5(args.h5)
    validate_dataset(args.h5, args.batch_size)


if __name__ == "__main__":
    main()
