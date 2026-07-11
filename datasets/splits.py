"""Deterministic episode-level dataset splits."""

import numpy as np


def make_episode_split(
    num_episodes: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint sorted train and validation episode indices."""
    if num_episodes < 2:
        raise ValueError("At least two episodes are required")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    num_val = max(1, round(num_episodes * val_fraction))
    num_val = min(num_val, num_episodes - 1)

    episode_indices = np.arange(num_episodes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_indices)

    val_indices = np.sort(episode_indices[:num_val])
    train_indices = np.sort(episode_indices[num_val:])
    return train_indices, val_indices
