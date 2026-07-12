"""Render real and reconstructed validation frames side by side."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.smb_dataset import SMBSubTrajectoryDataset
from datasets.splits import make_episode_split
from models.decoder import PixelDecoder
from models.jepa import Jepa


def to_image(tensor: torch.Tensor) -> Image.Image:
    pixels = tensor.detach().float().clamp(-1, 1)
    pixels = ((pixels + 1.0) * 127.5).byte()
    array = pixels.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    with h5py.File(args.h5, "r") as h5:
        num_episodes = len(h5["episodes"])
    _, val_episode_indices = make_episode_split(
        num_episodes,
        val_fraction=0.1,
        seed=args.split_seed,
    )
    dataset = SMBSubTrajectoryDataset(
        args.h5,
        episode_indices=val_episode_indices,
    )

    rng = np.random.default_rng(0)
    sample_indices = rng.choice(len(dataset), size=args.samples, replace=False)
    real_frames = torch.stack(
        [dataset[int(index)]["frames"][0] for index in sample_indices]
    ).to(device)

    jepa = Jepa().to(device)
    jepa_state = torch.load(
        args.jepa_checkpoint,
        map_location=device,
        weights_only=False,
    )
    jepa.load_state_dict(jepa_state["jepa"])
    jepa.eval()

    decoder = PixelDecoder().to(device)
    decoder_state = torch.load(
        args.decoder_checkpoint,
        map_location=device,
        weights_only=False,
    )
    decoder.load_state_dict(decoder_state["decoder"])
    decoder.eval()

    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    with torch.inference_mode(), precision_context:
        latent = jepa.encoder(real_frames)
        reconstructions = decoder(latent)

    header_height = 28
    row_gap = 4
    frame_size = 224
    canvas = Image.new(
        "RGB",
        (
            frame_size * 2,
            header_height + args.samples * frame_size + (args.samples - 1) * row_gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((82, 7), "REAL", fill="black")
    draw.text((294, 7), "RECONSTRUCTION", fill="black")

    for row in range(args.samples):
        y = header_height + row * (frame_size + row_gap)
        canvas.paste(to_image(real_frames[row]), (0, y))
        canvas.paste(to_image(reconstructions[row]), (frame_size, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    dataset.close()
    print(f"Saved validation reconstruction grid to {args.output}")


if __name__ == "__main__":
    main()
