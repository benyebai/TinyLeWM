"""Check checkpoint contents and atomic replacement using a tiny model."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from models.sigreg import SigReg
from scripts.train_jepa import get_lr_scheduler, load_checkpoint, save_checkpoint


def main() -> None:
    model = nn.Linear(3, 2)
    sigreg = SigReg(num_arrows=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_lr_scheduler(optimizer, total_steps=10, warmup_steps=1)
    saved_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "checkpoint.pt"
        save_checkpoint(
            path,
            model,
            sigreg,
            optimizer,
            scheduler,
            epoch=2,
            global_step=7,
            config={"batch_size": 128},
        )

        assert path.exists()
        assert not path.with_suffix(path.suffix + ".tmp").exists()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 2
        assert checkpoint["global_step"] == 7
        assert checkpoint["config"] == {"batch_size": 128}
        assert set(checkpoint) == {
            "epoch",
            "global_step",
            "config",
            "jepa",
            "sigreg",
            "optimizer",
            "scheduler",
        }

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(100.0)
        optimizer.param_groups[0]["lr"] = 0.9

        loaded = load_checkpoint(
            path,
            model,
            sigreg,
            optimizer,
            scheduler,
            device="cpu",
        )

        for parameter, saved_parameter in zip(model.parameters(), saved_parameters):
            assert torch.equal(parameter, saved_parameter)
        assert optimizer.param_groups[0]["lr"] != 0.9
        assert loaded["epoch"] == 2
        assert loaded["global_step"] == 7

    print("Checkpoint save/load ok")


if __name__ == "__main__":
    main()
