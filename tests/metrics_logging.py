"""Check that JSONL metric records append without overwriting earlier steps."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.train_jepa import append_jsonl


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "logs" / "metrics.jsonl"
        append_jsonl(path, {"step": 1, "loss": 1.5})
        append_jsonl(path, {"step": 2, "loss": 1.0})

        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records == [
            {"step": 1, "loss": 1.5},
            {"step": 2, "loss": 1.0},
        ]

    print("Metrics logging ok")


if __name__ == "__main__":
    main()
