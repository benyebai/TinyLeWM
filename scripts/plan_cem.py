"""Categorical CEM planning in LeMario's latent space."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from models.jepa import Jepa


ACTION_NAMES = [
    "NOOP",
    "RIGHT",
    "RIGHT+A",
    "RIGHT+B",
    "RIGHT+A+B",
    "A",
    "LEFT",
    "LEFT+A",
    "LEFT+B",
    "LEFT+A+B",
    "DOWN",
    "UP",
]

# Model slot order: [Left, Right, Up, Down, A, B]. This order matches the
# Gym's COMPLEX_MOVEMENT category order above.
ACTION_BITS = torch.tensor(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ],
    dtype=torch.float32,
)


def rollout(
    jepa: Jepa,
    state_history: torch.Tensor,
    past_action_history: torch.Tensor,
    future_action_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Roll forward from three states using five future action embeddings."""
    history = state_history
    action_history = past_action_history
    for step in range(future_action_embeddings.size(1)):
        next_action = future_action_embeddings[:, step : step + 1]
        conditions = torch.cat((action_history, next_action), dim=1)
        next_state = jepa.predict(history, conditions)[:, -1:]
        history = torch.cat((history[:, 1:], next_state), dim=1)
        action_history = torch.cat((action_history[:, 1:], next_action), dim=1)
    return history[:, -1]


def categories_to_action_embeddings(
    jepa: Jepa,
    categories: torch.Tensor,
) -> torch.Tensor:
    action_bits = ACTION_BITS.to(categories.device)[categories]
    action_blocks = action_bits.unsqueeze(2).expand(-1, -1, 5, -1)
    return jepa.action_encoder(action_blocks.flatten(start_dim=2))


def cem_plan(
    jepa: Jepa,
    state_history: torch.Tensor,
    past_action_history: torch.Tensor,
    goal: torch.Tensor,
    population: int = 300,
    elites: int = 30,
    iterations: int = 10,
    horizon: int = 5,
    score_function=None,
) -> tuple[torch.Tensor, float, float]:
    device = state_history.device
    num_actions = len(ACTION_NAMES)
    probabilities = torch.full(
        (horizon, num_actions),
        1.0 / num_actions,
        device=device,
    )
    initial_mean_score = 0.0
    best_categories = None
    best_score = float("inf")

    for iteration in range(iterations):
        categories = torch.multinomial(
            probabilities,
            num_samples=population,
            replacement=True,
        ).transpose(0, 1)
        future_actions = categories_to_action_embeddings(jepa, categories)
        final_states = rollout(
            jepa,
            state_history.expand(population, -1, -1),
            past_action_history.expand(population, -1, -1),
            future_actions,
        )
        scores = (
            score_function(final_states)
            if score_function is not None
            else (final_states - goal).square().mean(dim=-1)
        )

        if iteration == 0:
            initial_mean_score = scores.mean().item()

        elite_indices = scores.topk(elites, largest=False).indices
        elite_categories = categories[elite_indices]
        elite_probabilities = torch.nn.functional.one_hot(
            elite_categories,
            num_classes=num_actions,
        ).float().mean(dim=0)
        probabilities = elite_probabilities.clamp_min(0.01)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

        iteration_best = scores.argmin()
        if scores[iteration_best].item() < best_score:
            best_score = scores[iteration_best].item()
            best_categories = categories[iteration_best].clone()

    assert best_categories is not None
    return best_categories, best_score, initial_mean_score


def main() -> None:
    import h5py

    from datasets.smb_dataset import SMBSubTrajectoryDataset
    from datasets.splits import make_episode_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with h5py.File(args.h5, "r") as h5:
        num_episodes = len(h5["episodes"])
    _, val_indices = make_episode_split(num_episodes, 0.1, 42)
    dataset = SMBSubTrajectoryDataset(
        args.h5,
        num_frames=9,
        episode_indices=val_indices,
    )
    sample = dataset[args.sample_index % len(dataset)]
    frames = sample["frames"].unsqueeze(0).to(device)
    actions = sample["actions"].unsqueeze(0).to(device)

    jepa = Jepa().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    jepa.load_state_dict(checkpoint["jepa"])
    jepa.eval()

    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    with torch.inference_mode(), precision_context:
        embeddings, action_embeddings = jepa.encode(frames, actions)
        state_history = embeddings[:, 1:4]
        past_action_history = action_embeddings[:, 1:3]
        goal = embeddings[:, 8]

        planned, planned_score, random_mean_score = cem_plan(
            jepa,
            state_history,
            past_action_history,
            goal,
        )
        recorded_final = rollout(
            jepa,
            state_history,
            past_action_history,
            action_embeddings[:, 3:8],
        )
        recorded_score = (recorded_final - goal).square().mean().item()

    dataset.close()
    improvement = 100.0 * (1.0 - planned_score / random_mean_score)
    print(f"checkpoint_step={checkpoint['global_step']}")
    print(f"planned actions: {[ACTION_NAMES[index] for index in planned.tolist()]}")
    print(f"CEM predicted goal distance:      {planned_score:.6f}")
    print(f"random-sequence mean distance:    {random_mean_score:.6f}")
    print(f"recorded-action predicted distance: {recorded_score:.6f}")
    print(f"CEM improvement over random:      {improvement:.1f}%")


if __name__ == "__main__":
    main()
