# LeMario

LeMario is a compact, from-scratch implementation of an action-conditioned
joint-embedding predictive architecture for Super Mario Bros. It is inspired by
[LeWorldModel](https://arxiv.org/abs/2603.19312).

The model learns short-horizon latent dynamics from frames and controller
inputs. It is a research prototype, not a general Mario-playing agent.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.example
```

## Data and training

Convert a local `data-smb/` recording set to HDF5:

```bash
python -m datasets.build_hdf5 --src data-smb --out data/smb.h5
```

Train and evaluate:

```bash
python -m scripts.train_jepa --h5 data/smb.h5 --output-dir checkpoints
python -m scripts.eval_jepa --h5 data/smb.h5 --checkpoint checkpoints/jepa_final.pt
python -m scripts.eval_rollout --h5 data/smb.h5 --checkpoint checkpoints/jepa_final.pt
```

Run latent-space CEM planning:

```bash
python -m scripts.plan_cem --h5 data/smb.h5 --checkpoint checkpoints/jepa_final.pt
```

## Repository

```text
datasets/  data conversion and loading
models/    encoder, action encoder, predictor, and SIGReg
scripts/   example, training, evaluation, and planning
tests/     small architecture checks
```

Datasets, checkpoints, experiments, notes, and media are intentionally kept out
of the public repository.
