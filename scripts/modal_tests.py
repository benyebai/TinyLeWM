"""Modal entrypoints for TinyLeWM's overfit and smoke-test gates."""

import subprocess
import sys
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_PROJECT = "/root/project"
REMOTE_H5 = "/root/data/smb.h5"
VOLUME_ROOT = "/storage"
FULL_H5 = f"{VOLUME_ROOT}/data/smb-full.h5"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers>=4.40,<5",
        "einops>=0.7,<1",
        "h5py>=3.10,<4",
        "numpy>=1.26,<3",
        "opencv-python-headless>=4.9,<5",
        "pillow>=10,<13",
    )
    .add_local_dir(PROJECT_ROOT / "datasets", f"{REMOTE_PROJECT}/datasets")
    .add_local_dir(PROJECT_ROOT / "models", f"{REMOTE_PROJECT}/models")
    .add_local_dir(PROJECT_ROOT / "scripts", f"{REMOTE_PROJECT}/scripts")
    .add_local_dir(PROJECT_ROOT / "tests", f"{REMOTE_PROJECT}/tests")
    .add_local_dir(PROJECT_ROOT / "utils", f"{REMOTE_PROJECT}/utils")
    .add_local_file(PROJECT_ROOT / "data/smb.h5", REMOTE_H5)
)

app = modal.App("tinylewm-test-gates")
storage = modal.Volume.from_name("tinylewm-storage")


def run_project_module(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    subprocess.run(command, cwd=REMOTE_PROJECT, check=True)


@app.function(
    image=image,
    volumes={VOLUME_ROOT: storage},
    timeout=10 * 60,
)
def validate_full_data() -> None:
    run_project_module(
        "tests.validate_smb_dataset",
        ["--h5", FULL_H5, "--batch-size", "8"],
    )


@app.function(image=image, gpu="A10", timeout=30 * 60)
def run_overfit(steps: int = 300) -> None:
    run_project_module(
        "scripts.overfit_jepa",
        ["--h5", REMOTE_H5, "--samples", "10", "--steps", str(steps)],
    )


@app.function(image=image, gpu="A10", timeout=30 * 60)
def run_smoke(steps: int = 200, batch_size: int = 16) -> None:
    run_project_module(
        "scripts.smoke_jepa",
        [
            "--h5",
            REMOTE_H5,
            "--steps",
            str(steps),
            "--batch-size",
            str(batch_size),
        ],
    )
