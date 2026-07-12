"""Modal entrypoints for TinyLeWM's overfit and smoke-test gates."""

import os
import shutil
import subprocess
import sys
import time
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

mario_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers>=4.40,<5",
        "einops>=0.7,<1",
        "numpy>=2,<3",
        "opencv-python-headless>=4.9,<5",
        "pillow>=10,<13",
        "gym-super-mario-bros==9.1.0",
    )
    .add_local_dir(PROJECT_ROOT / "models", f"{REMOTE_PROJECT}/models")
    .add_local_dir(PROJECT_ROOT / "scripts", f"{REMOTE_PROJECT}/scripts")
)

app = modal.App("tinylewm-test-gates")
storage = modal.Volume.from_name("tinylewm-storage")


def run_project_module(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(
        command,
        cwd=REMOTE_PROJECT,
        check=True,
        env=environment,
    )


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
def run_overfit(
    steps: int = 300,
    samples: int = 10,
    target_pred: float = 0.01,
) -> None:
    run_project_module(
        "scripts.overfit_jepa",
        [
            "--h5",
            REMOTE_H5,
            "--samples",
            str(samples),
            "--steps",
            str(steps),
            "--target-pred",
            str(target_pred),
        ],
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


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=32_768,
    volumes={VOLUME_ROOT: storage},
    timeout=12 * 60 * 60,
)
def train_full_data(
    max_steps: int = 500,
    batch_size: int = 128,
    run_name: str = "full-data-gate",
    num_workers: int = 8,
    log_every: int = 10,
    checkpoint_every: int = 100,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    copy_started = time.monotonic()
    print(f"Copying {FULL_H5} to local container storage...", flush=True)
    shutil.copyfile(FULL_H5, local_h5)
    print(
        f"Dataset copy finished in {time.monotonic() - copy_started:.1f}s",
        flush=True,
    )

    output_dir = f"{VOLUME_ROOT}/runs/{run_name}"
    run_project_module(
        "scripts.train_jepa",
        [
            "--h5",
            local_h5,
            "--output-dir",
            output_dir,
            "--batch-size",
            str(batch_size),
            "--num-epochs",
            "10",
            "--max-steps",
            str(max_steps),
            "--log-every",
            str(log_every),
            "--checkpoint-every",
            str(checkpoint_every),
            "--num-workers",
            str(num_workers),
        ],
    )
    storage.commit()


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32_768,
    volumes={VOLUME_ROOT: storage},
    timeout=30 * 60,
)
def evaluate_jepa(
    run_name: str = "jepa-one-epoch-v1",
    batch_size: int = 128,
    max_batches: int = 100,
    num_workers: int = 8,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_checkpoint = "/tmp/jepa_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{run_name}/jepa_final.pt",
        local_checkpoint,
    )
    run_project_module(
        "scripts.eval_jepa",
        [
            "--h5",
            local_h5,
            "--checkpoint",
            local_checkpoint,
            "--batch-size",
            str(batch_size),
            "--max-batches",
            str(max_batches),
            "--num-workers",
            str(num_workers),
        ],
    )


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32_768,
    volumes={VOLUME_ROOT: storage},
    timeout=30 * 60,
)
def evaluate_rollout(
    run_name: str = "jepa-one-epoch-v1",
    batch_size: int = 32,
    max_batches: int = 100,
    num_workers: int = 8,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_checkpoint = "/tmp/jepa_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{run_name}/jepa_final.pt",
        local_checkpoint,
    )
    run_project_module(
        "scripts.eval_rollout",
        [
            "--h5",
            local_h5,
            "--checkpoint",
            local_checkpoint,
            "--batch-size",
            str(batch_size),
            "--max-batches",
            str(max_batches),
            "--num-workers",
            str(num_workers),
        ],
    )


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32_768,
    volumes={VOLUME_ROOT: storage},
    timeout=30 * 60,
)
def run_decoder_overfit(
    run_name: str = "jepa-one-epoch-v1",
    steps: int = 200,
    batch_size: int = 8,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_checkpoint = "/tmp/jepa_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{run_name}/jepa_final.pt",
        local_checkpoint,
    )
    run_project_module(
        "scripts.train_decoder",
        [
            "--h5",
            local_h5,
            "--jepa-checkpoint",
            local_checkpoint,
            "--output-dir",
            f"{VOLUME_ROOT}/runs/decoder-overfit-v2",
            "--batch-size",
            str(batch_size),
            "--max-steps",
            str(steps),
            "--num-workers",
            "8",
            "--log-every",
            "20",
            "--checkpoint-every",
            str(steps),
            "--fixed-batch",
            "--preview-output",
            f"{VOLUME_ROOT}/runs/decoder-overfit-v2/preview.png",
        ],
    )
    storage.commit()


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=32_768,
    volumes={VOLUME_ROOT: storage},
    timeout=6 * 60 * 60,
)
def train_decoder_full(
    jepa_run_name: str = "jepa-one-epoch-v1",
    decoder_run_name: str = "decoder-v1",
    max_steps: int = 5000,
    batch_size: int = 32,
    num_workers: int = 8,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_checkpoint = "/tmp/jepa_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{jepa_run_name}/jepa_final.pt",
        local_checkpoint,
    )
    run_project_module(
        "scripts.train_decoder",
        [
            "--h5",
            local_h5,
            "--jepa-checkpoint",
            local_checkpoint,
            "--output-dir",
            f"{VOLUME_ROOT}/runs/{decoder_run_name}",
            "--batch-size",
            str(batch_size),
            "--max-steps",
            str(max_steps),
            "--num-workers",
            str(num_workers),
            "--log-every",
            "100",
            "--checkpoint-every",
            "1000",
        ],
    )
    storage.commit()


@app.function(
    image=image,
    gpu="A10",
    volumes={VOLUME_ROOT: storage},
    timeout=20 * 60,
)
def render_decoder_samples(
    jepa_run_name: str = "jepa-one-epoch-v1",
    decoder_run_name: str = "decoder-v1",
    samples: int = 6,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_jepa = "/tmp/jepa_final.pt"
    local_decoder = "/tmp/decoder_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{jepa_run_name}/jepa_final.pt",
        local_jepa,
    )
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{decoder_run_name}/decoder_final.pt",
        local_decoder,
    )
    run_project_module(
        "scripts.render_decoder_samples",
        [
            "--h5",
            local_h5,
            "--jepa-checkpoint",
            local_jepa,
            "--decoder-checkpoint",
            local_decoder,
            "--output",
            f"{VOLUME_ROOT}/runs/{decoder_run_name}/reconstruction_grid.png",
            "--samples",
            str(samples),
        ],
    )
    storage.commit()


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=32_768,
    volumes={VOLUME_ROOT: storage},
    timeout=3 * 60 * 60,
)
def finetune_decoder_details(
    jepa_run_name: str = "jepa-one-epoch-v1",
    input_decoder_run_name: str = "decoder-v2",
    output_decoder_run_name: str = "decoder-v3",
    max_steps: int = 2000,
    batch_size: int = 32,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_jepa = "/tmp/jepa_final.pt"
    local_decoder = "/tmp/decoder_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{jepa_run_name}/jepa_final.pt",
        local_jepa,
    )
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{input_decoder_run_name}/decoder_final.pt",
        local_decoder,
    )
    run_project_module(
        "scripts.train_decoder",
        [
            "--h5",
            local_h5,
            "--jepa-checkpoint",
            local_jepa,
            "--decoder-checkpoint",
            local_decoder,
            "--output-dir",
            f"{VOLUME_ROOT}/runs/{output_decoder_run_name}",
            "--batch-size",
            str(batch_size),
            "--max-steps",
            str(max_steps),
            "--learning-rate",
            "5e-5",
            "--num-workers",
            "8",
            "--log-every",
            "100",
            "--checkpoint-every",
            "1000",
            "--hard-pixel-weight",
            "0.25",
            "--edge-weight",
            "0.1",
            "--hard-fraction",
            "0.05",
        ],
    )
    storage.commit()


@app.function(
    image=image,
    gpu="A10",
    volumes={VOLUME_ROOT: storage},
    timeout=20 * 60,
)
def test_cem(
    run_name: str = "jepa-one-epoch-v1",
    sample_index: int = 1000,
) -> None:
    local_h5 = "/tmp/smb-full.h5"
    local_checkpoint = "/tmp/jepa_final.pt"
    shutil.copyfile(FULL_H5, local_h5)
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{run_name}/jepa_final.pt",
        local_checkpoint,
    )
    run_project_module(
        "scripts.plan_cem",
        [
            "--h5",
            local_h5,
            "--checkpoint",
            local_checkpoint,
            "--sample-index",
            str(sample_index),
        ],
    )


@app.function(
    image=mario_image,
    gpu="A10",
    volumes={VOLUME_ROOT: storage},
    timeout=30 * 60,
)
def run_live_cem(
    run_name: str = "jepa-one-epoch-v1",
    output_name: str = "live-cem-v1",
) -> None:
    local_checkpoint = "/tmp/jepa_final.pt"
    shutil.copyfile(
        f"{VOLUME_ROOT}/runs/{run_name}/jepa_final.pt",
        local_checkpoint,
    )
    run_project_module(
        "scripts.live_cem_mario",
        [
            "--checkpoint",
            local_checkpoint,
            "--output-dir",
            f"{VOLUME_ROOT}/runs/{output_name}",
        ],
    )
    storage.commit()
