"""
PyTorch Dataset for SMB sub-trajectories stored in HDF5.

The HDF5 file is produced by `datasets.build_hdf5` and contains:
    /frames    uint8 [N, 240, 256, 3]
    /actions   uint8 [N, 6]
    /episodes  int64 [E, 2] where each row is (start_idx, length)

Each sample returns:
    frames:  float32 [num_frames, 3, image_size, image_size], range [-1, 1]
    actions: float32 [num_frames, frame_skip, 6], values {0, 1}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-untyped]
import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch.utils.data import Dataset


class SMBSubTrajectoryDataset(Dataset):
    """Sliding-window sub-trajectories from the SMB HDF5 dataset."""

    def __init__(
        self,
        h5_path: str | Path,
        num_frames: int = 4,
        frame_skip: int = 5,
        image_size: int = 224,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.image_size = image_size
        self._h5: Any | None = None

        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if image_size < 1:
            raise ValueError("image_size must be at least 1")

        with h5py.File(self.h5_path, "r") as h5:
            self.starts = self._build_valid_starts(h5["episodes"][:])

    def _build_valid_starts(self, episodes: np.ndarray) -> np.ndarray:
        """Return global frame indices whose full sample stays inside an episode."""
        valid_starts = []
        needed_span = self.num_frames * self.frame_skip

        # builds [[], []] where each sublist has a bunch of valid starts for that episode (ep determined by index)
        for ep_start, ep_len in episodes:
            # Need observed frames [s, s+5, s+10, s+15] plus action block
            # actions[s+15:s+20], so the sample consumes 20 source indices.
            n = int(ep_len) - needed_span + 1
            if n <= 0:
                continue
            ep_start = int(ep_start)
            valid_starts.append(np.arange(ep_start, ep_start + n, dtype=np.int64))

        if not valid_starts:
            raise ValueError("No valid sub-trajectories found in HDF5 episodes")

        # squashes the valid starts into one array
        return np.concatenate(valid_starts)

    def _get_h5(self) -> Any:
        """Open HDF5 lazily once per DataLoader worker."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5 = self._get_h5()
        start = int(self.starts[idx])

        # using the random start that we have, it will jsut get [f1: 5 frames], [f2: 5 frames]...
        # and the 20 actions, 5 actions for each f
        #
        # returns them in a nice json

        frame_indices = start + np.arange(self.num_frames) * self.frame_skip
        action_start = start
        action_end = start + self.num_frames * self.frame_skip

        frames = h5["frames"][frame_indices]
        actions = h5["actions"][action_start:action_end]

        frames = self._preprocess_frames(frames)
        actions = actions.reshape(self.num_frames, self.frame_skip, 6).astype(
            np.float32
        )

        return {
            "frames": torch.from_numpy(frames),
            "actions": torch.from_numpy(actions),
            "start_idx": torch.tensor(start, dtype=torch.long),
        }

    def _preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        resized = np.empty(
            (self.num_frames, self.image_size, self.image_size, 3),
            dtype=np.float32,
        )

        for i, frame in enumerate(frames):
            img = cv2.resize(
                frame,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            resized[i] = (img.astype(np.float32) / 127.5) - 1.0

        return np.transpose(resized, (0, 3, 1, 2))

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()
