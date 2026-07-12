"""Run TinyLeWM CEM actions in the real Mario emulator."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import cv2
import gymnasium as gym
import gym_super_mario_bros  # noqa: F401 - registers Mario environments
import numpy as np
import torch
from nes_py.wrappers import JoypadSpace
from PIL import Image, ImageDraw

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from models.jepa import Jepa
from scripts.plan_cem import ACTION_BITS, ACTION_NAMES, cem_plan


def preprocess(observation: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(observation, (224, 224), interpolation=cv2.INTER_AREA)
    frame = resized.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(frame.transpose(2, 0, 1))


def step_block(env, action: int, video_frames: list[np.ndarray] | None = None):
    observation = None
    info = None
    done = False
    for _ in range(5):
        observation, _, terminated, truncated, info = env.step(action)
        if video_frames is not None:
            video_frames.append(observation.copy())
        done = bool(terminated or truncated)
        if done:
            break
    return observation, info, done


def establish_history(env, seed: int):
    observation, _ = env.reset(seed=seed)
    observations = [observation.copy()]
    infos = []
    for _ in range(2):
        observation, info, done = step_block(env, 0)
        if done:
            raise RuntimeError("Mario terminated while establishing history")
        observations.append(observation.copy())
        infos.append(info)
    return observations, infos[-1]


def execute_sequence(env, actions: list[int], record: bool = False):
    frames = [] if record else None
    observation = None
    info = None
    done = False
    for action in actions:
        observation, info, done = step_block(env, action, frames)
        if done:
            break
    return observation, info, done, frames


def save_result_grid(
    initial: np.ndarray,
    goal: np.ndarray,
    final: np.ndarray,
    output: Path,
) -> None:
    images = [Image.fromarray(frame) for frame in (initial, goal, final)]
    labels = ["START", "GOAL", "CEM ACTUAL"]
    width, height = images[0].size
    header = 28
    canvas = Image.new("RGB", (width * 3, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index * width
        canvas.paste(image, (x, header))
        draw.text((x + 8, 7), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    jepa = Jepa().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    jepa.load_state_dict(checkpoint["jepa"])
    jepa.eval()

    env = gym.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    # Build an attainable local goal: move right while running for 25 frames.
    reference_actions = [3] * 5  # COMPLEX_MOVEMENT index 3 = RIGHT+B
    reference_history, reference_start_info = establish_history(env, args.seed)
    goal_observation, goal_info, goal_done, _ = execute_sequence(
        env,
        reference_actions,
    )
    if goal_done:
        raise RuntimeError("Reference goal terminated unexpectedly")

    history_observations, start_info = establish_history(env, args.seed)
    history_tensor = torch.stack(
        [preprocess(frame) for frame in history_observations]
    ).to(device)
    goal_tensor = preprocess(goal_observation).unsqueeze(0).to(device)

    with torch.inference_mode(), precision_context:
        state_history = jepa.encoder(history_tensor).unsqueeze(0)
        goal_latent = jepa.encoder(goal_tensor)
        noop_blocks = ACTION_BITS[0].reshape(1, 1, 1, 6).expand(1, 2, 5, 6)
        noop_blocks = noop_blocks.to(device).flatten(start_dim=2)
        past_action_history = jepa.action_encoder(noop_blocks)
        planned, predicted_score, random_predicted_mean = cem_plan(
            jepa,
            state_history,
            past_action_history,
            goal_latent,
        )

    planned_list = planned.tolist()
    final_observation, final_info, planned_done, video_frames = execute_sequence(
        env,
        planned_list,
        record=True,
    )

    random_generator = torch.Generator().manual_seed(123)
    random_sequences = torch.randint(
        0,
        len(ACTION_NAMES),
        (args.random_trials, 5),
        generator=random_generator,
    )
    random_final_observations = []
    random_x_positions = []
    for sequence in random_sequences.tolist():
        establish_history(env, args.seed)
        random_final, random_info, _, _ = execute_sequence(env, sequence)
        random_final_observations.append(preprocess(random_final))
        random_x_positions.append(random_info["x_pos"])

    final_tensor = preprocess(final_observation).unsqueeze(0).to(device)
    random_tensor = torch.stack(random_final_observations).to(device)
    with torch.inference_mode(), precision_context:
        final_latent = jepa.encoder(final_tensor)
        random_latents = jepa.encoder(random_tensor)
        actual_distance = (final_latent - goal_latent).square().mean().item()
        random_actual_distances = (
            (random_latents - goal_latent).square().mean(dim=-1).float()
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "live_cem_result.png"
    gif_path = args.output_dir / "live_cem_rollout.gif"
    save_result_grid(
        history_observations[-1],
        goal_observation,
        final_observation,
        result_path,
    )
    gif_images = [Image.fromarray(history_observations[-1])]
    gif_images.extend(Image.fromarray(frame) for frame in video_frames)
    gif_images[0].save(
        gif_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=50,
        loop=0,
    )
    env.close()

    print(f"checkpoint_step={checkpoint['global_step']}")
    print(f"planned actions: {[ACTION_NAMES[index] for index in planned_list]}")
    print(f"predicted goal distance:   {predicted_score:.6f}")
    print(f"predicted random mean:     {random_predicted_mean:.6f}")
    print(f"actual goal distance:      {actual_distance:.6f}")
    print(f"actual random mean:        {random_actual_distances.mean().item():.6f}")
    print(f"start x_pos: {start_info['x_pos']}")
    print(f"goal x_pos:  {goal_info['x_pos']}")
    print(f"CEM x_pos:   {final_info['x_pos']}")
    print(f"random mean x_pos: {sum(random_x_positions) / len(random_x_positions):.1f}")
    print(f"CEM terminated: {planned_done}")
    print(f"saved result: {result_path}")
    print(f"saved GIF:    {gif_path}")


if __name__ == "__main__":
    main()
