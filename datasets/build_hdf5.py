"""
Build HDF5 from data-smb/ PNG dataset.

Walks every episode directory, parses filenames, decodes PNGs to uint8 RGB,
and writes one consolidated HDF5 file with four datasets:
    /frames          uint8 [N, 240, 256, 3]   gzip-4, chunked by 64 frames
    /actions         uint8 [N, 6]             6-bit multi-hot [L,R,U,D,A,B]
    /frame_metadata  struct [N]               (player, sessid, episode, level, frame_idx, outcome)
    /episodes        i8    [E, 2]             (start_idx, length)

Usage:
    python -m datasets.build_hdf5 \\
        --src "/path/to/data-smb" \\
        --out data/smb.h5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# Make `utils.action_codes` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.action_codes import action_code_to_multihot, action_codes_to_multihot

# PIL gives H x W x C, and the SMB PNGs are 240 high x 256 wide.
FRAME_H, FRAME_W, FRAME_C = 240, 256, 3

# {player}_{sessid}_e{ep}_{level}_f{frame}_a{action}_{date}_{time}.{outcome}.png
FNAME_RE = re.compile(
    r"^(?P<player>[^_]+)_(?P<sessid>[^_]+)_e(?P<episode>\d+)_(?P<level>\d+-\d+)_"
    r"f(?P<frame_idx>\d+)_a(?P<action>\d+)_"
    r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.(?P<outcome>fail|win)\.png$"
)

META_DTYPE = np.dtype(
    [
        ("player_id", "S32"),
        ("sessid", "S32"),
        ("episode", "i4"),
        ("level", "S8"),
        ("frame_idx", "i4"),
        ("outcome", "S8"),
    ]
)


def parse_filename(fname: str) -> dict:
    m = FNAME_RE.match(fname)
    if m is None:
        raise ValueError(f"Unrecognized filename: {fname}")
    return m.groupdict()


def index_dataset(src: Path) -> list[dict]:
    """Walk data-smb/, return one entry per episode with frames sorted by frame_idx."""
    episodes = []
    for ep_dir in sorted(src.iterdir()):
        if not ep_dir.is_dir():
            continue
        frames = []
        for fname in os.listdir(ep_dir):
            if not fname.endswith(".png"):
                continue
            meta = parse_filename(fname)
            frames.append(
                {
                    "path": ep_dir / fname,
                    "player": meta["player"],
                    "sessid": meta["sessid"],
                    "episode": int(meta["episode"]),
                    "level": meta["level"],
                    "frame_idx": int(meta["frame_idx"]),
                    "action_code": int(meta["action"]),
                    "outcome": meta["outcome"],
                }
            )
        if not frames:
            continue
        frames.sort(key=lambda f: f["frame_idx"])
        first = frames[0]
        episodes.append(
            {
                "key": (first["player"], first["sessid"], first["episode"]),
                "length": len(frames),
                "frames": frames,
            }
        )
    episodes.sort(key=lambda e: e["key"])
    return episodes


def build(src: Path, out: Path, gzip_level: int = 4, chunk_frames: int = 64) -> None:
    print(f"Indexing {src} ...")
    episodes = index_dataset(src)
    total_frames = sum(e["length"] for e in episodes)
    n_episodes = len(episodes)
    print(f"Found {n_episodes} episodes, {total_frames} frames total")

    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as h5:
        frames_ds = h5.create_dataset(
            "frames",
            shape=(total_frames, FRAME_H, FRAME_W, FRAME_C),
            dtype=np.uint8,
            chunks=(chunk_frames, FRAME_H, FRAME_W, FRAME_C),
            compression="gzip",
            compression_opts=gzip_level,
        )
        actions_ds = h5.create_dataset(
            "actions", shape=(total_frames, 6), dtype=np.uint8
        )
        meta_ds = h5.create_dataset(
            "frame_metadata", shape=(total_frames,), dtype=META_DTYPE
        )
        ep_ds = h5.create_dataset("episodes", shape=(n_episodes, 2), dtype=np.int64)

        # Buffer one chunk of frames so each gzip write covers a full chunk.
        buf = np.empty((chunk_frames, FRAME_H, FRAME_W, FRAME_C), dtype=np.uint8)
        buf_n = 0
        buf_start = 0

        global_idx = 0
        spot_checks_left = {100, 101}
        for ep_i, ep in enumerate(tqdm(episodes, desc="episodes")):
            ep_start = global_idx
            ep_len = ep["length"]
            ep_ds[ep_i] = (ep_start, ep_len)

            action_codes = np.array(
                [f["action_code"] for f in ep["frames"]], dtype=np.int64
            )
            actions_ds[ep_start : ep_start + ep_len] = action_codes_to_multihot(
                action_codes
            )

            meta_rows = np.empty(ep_len, dtype=META_DTYPE)
            for i, f in enumerate(ep["frames"]):
                meta_rows[i] = (
                    f["player"].encode(),
                    f["sessid"].encode(),
                    f["episode"],
                    f["level"].encode(),
                    f["frame_idx"],
                    f["outcome"].encode(),
                )
            meta_ds[ep_start : ep_start + ep_len] = meta_rows

            for f in ep["frames"]:
                img = Image.open(f["path"]).convert("RGB")
                arr = np.asarray(img, dtype=np.uint8)
                if arr.shape != (FRAME_H, FRAME_W, FRAME_C):
                    raise ValueError(f"Unexpected shape {arr.shape} at {f['path']}")

                buf[buf_n] = arr
                buf_n += 1
                if buf_n == chunk_frames:
                    frames_ds[buf_start : buf_start + buf_n] = buf[:buf_n]
                    buf_start += buf_n
                    buf_n = 0

                if ep_i == 0 and f["frame_idx"] in spot_checks_left:
                    spot_checks_left.discard(f["frame_idx"])
                    mh = action_code_to_multihot(f["action_code"])
                    print(
                        f"  spot: ep0 frame_idx={f['frame_idx']} "
                        f"action_code={f['action_code']} multihot={mh.tolist()} "
                        f"pixel_mean={arr.mean():.2f}"
                    )

                global_idx += 1

        if buf_n > 0:
            frames_ds[buf_start : buf_start + buf_n] = buf[:buf_n]

        assert global_idx == total_frames, (global_idx, total_frames)

    print(f"Wrote {out} — {total_frames} frames, {n_episodes} episodes")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Path to data-smb/ root")
    p.add_argument("--out", type=Path, default=Path("data/smb.h5"))
    p.add_argument("--gzip", type=int, default=4)
    p.add_argument("--chunk-frames", type=int, default=64)
    args = p.parse_args()
    build(args.src, args.out, gzip_level=args.gzip, chunk_frames=args.chunk_frames)


if __name__ == "__main__":
    main()
