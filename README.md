# TinyLeWM

TinyLeWM is a small, from-scratch reimplementation of the core ideas in
[LeWorldModel](https://arxiv.org/abs/2603.19312), trained on Super Mario Bros.
gameplay.

The model learns entirely from pixels and actions. An encoder turns each Mario
frame into a 192-dimensional latent state, and an action-conditioned predictor
tries to predict the next latent state. SIGReg keeps the encoder from mapping
every frame to the same representation.

## Current status

**The training and evaluation infrastructure works, and the model has learned
some action-conditioned short-horizon dynamics. It is not yet a successful
Mario controller.**

The current one-epoch checkpoint:

- predicts held-out next-frame latents slightly better than a no-change baseline;
- is sensitive to the supplied actions;
- beats a no-change baseline in an offline five-step rollout test;
- lets CEM optimize action sequences inside the learned model;
- but fails the first live goal-reaching test in the real NES emulator.

The live test exposed the present blocker: Euclidean distance in the learned
latent space does not reliably track Mario's position. Random actions that left
Mario near the start sometimes looked closer to the goal in latent space than
actions that moved him forward.

The next diagnostic is a frozen-latent position probe. We will collect frames
from the emulator with its true `x_pos` and `y_pos`, then test whether a small
linear model can recover Mario's position from the 192-dimensional latent.

## Model

### Inputs

Each training sample contains four observations separated by five emulator
frames:

```text
frames:  [B, 4, 3, 224, 224]
actions: [B, 4, 5, 6]
```

The six action slots are ordered as:

```text
[Left, Right, Up, Down, A, B]
```

Each `[5, 6]` block contains the controller states applied during the five NES
frames between two observations. It is flattened to 30 values before entering
the action encoder.

### Architecture

- **Visual encoder:** ViT operating on 224×224 RGB frames, with a
  192-dimensional output.
- **Action encoder:** a small MLP mapping 30 controller values to 192 values.
- **Predictor:** six causal action-conditioned transformer blocks using
  AdaLN-Zero conditioning.
- **Prediction projection:** maps predictor outputs back to the
  192-dimensional latent space.
- **SIGReg:** tests random one-dimensional projections of each timestep's
  embeddings against an isotropic Gaussian distribution.

### Training loss

For a predicted embedding `p` and the encoder's real future embedding `z`, the
prediction loss is mean squared error:

```text
prediction_loss = mean((p - z)²)
```

In code:

```python
pred_loss = F.mse_loss(next_emb[:, :-1], emb[:, 1:])
total_loss = pred_loss + 0.1 * sigreg_loss
```

Both sides of the prediction loss receive gradients. There is no target network,
EMA, or stop-gradient.

SIGReg does not have a magic target of exactly 1. Lower and bounded values are
generally better; its purpose is to prevent a collapsed representation.

### Important predictor detail

Four input frames produce four predictor outputs, but only the first three have
future targets inside the sample. The training loss therefore excludes the last
output with `next_emb[:, :-1]`.

Recursive evaluation must use the three trained context positions. An early
five-step evaluator accidentally used the untrained fourth output and produced a
meaningless rollout loss of `10.236848`. After correcting the context, the
five-step loss was `0.077717`.

## Dataset

The full local dataset was built from PNG gameplay recordings into
`data/smb-full.h5`.

### Raw and HDF5 counts

| Item | Value |
|---|---:|
| Raw frames | 737,134 |
| Episodes | 280 |
| HDF5 size | 2.4 GiB |
| Total valid four-frame windows | 731,814 |
| Frame shape | `(240, 256, 3)` `uint8` |
| Action shape | `(6,)` `uint8` |

HDF5 datasets:

```text
/frames         (737134, 240, 256, 3) uint8
/actions        (737134, 6)           uint8
/frame_metadata (737134,)
/episodes       280 rows
```

The PyTorch dataset resizes frames to 224×224 and normalizes them to `[-1, 1]`.

### Episode split

The split is deterministic, episode-level, and uses seed 42. Windows from one
episode can never appear in both partitions.

| Partition | Episodes | Windows | Levels | Outcomes |
|---|---:|---:|---:|---|
| Train | 252 | 655,923 | 32 | fail, win |
| Validation | 28 | 75,891 | 13 | fail, win |

All 13 validation levels also occur in training. The validation set does not
cover all 32 training levels, which is a known limitation.

## Training setup

Training ran on a Modal A100 using:

- BF16 autocast;
- batch size 128;
- AdamW, learning rate `3e-4`, weight decay `0.05`;
- 1,000-step linear warmup followed by cosine decay;
- gradient clipping at 1.0;
- eight data-loader workers;
- atomic checkpoints and JSONL metrics;
- a deterministic 90/10 episode split.

The full HDF5 is stored in the Modal Volume `tinylewm-storage` and copied to the
container's local disk at startup. This reduced the copy to roughly 2–4 seconds
and removed the original Volume I/O bottleneck.

### Completed training run

| Item | Value |
|---|---:|
| Run name | `jepa-one-epoch-v1` |
| Steps | 5,125 |
| Effective epochs | 1 |
| Training windows seen | approximately 655,923 |
| Wall time | approximately 1 hour 53 minutes |
| Final checkpoint size | 198.9 MiB |
| Device | Modal A100, BF16 |

The training log says `epoch=1/10` because the script allows ten epochs, but
`--max-steps 5125` intentionally stopped the run after one complete pass.

Loss snapshots:

| Step | Total loss | Prediction | SIGReg | Gradient norm |
|---:|---:|---:|---:|---:|
| 1 | 3.6676 | 1.1208 | 25.5000 | 55.7480 |
| 1,000 | 0.3173 | 0.0399 | 2.7812 | 0.6126 |
| 3,000 | 0.2435 | 0.0326 | 2.1094 | 0.4139 |
| 5,000 | 0.1942 | 0.0233 | 1.7109 | 0.3743 |
| 5,100 | 0.2026 | 0.0268 | 1.7578 | 0.3244 |

The final checkpoint records `global_step=5125`.

## Tests and benchmarks

### Infrastructure gates

| Test | Result |
|---|---|
| Full HDF5 structural validation | Pass |
| Dataset tensor shapes and ranges | Pass |
| Episode split determinism and leakage | Pass |
| Shared optimizer-step test | Pass |
| CPU float32 fallback | Pass |
| A10 BF16 execution | Pass |
| JSONL metrics logging | Pass |
| Atomic checkpoint save/load | Pass |
| Finite loss and gradient guards | Pass |

### JEPA overfit test

Ten fixed samples were repeated for 300 optimizer steps on an A10.

```text
first prediction loss: 0.1408
best prediction loss:  0.000321
reduction:              99.8%
target:                 < 0.01
result:                 PASS
```

This proves the model and optimizer can memorize a tiny dataset.

### JEPA smoke test

Two hundred varied-batch steps, batch size 16, on an A10:

```text
first-20 mean prediction loss: 0.717731
last-20 mean prediction loss:  0.046563
last-20 mean SIGReg loss:      1.936634
result:                        PASS
```

### Held-out one-step evaluation

The corrected evaluator randomly mixed windows from all 28 held-out episodes.
It evaluated 100 batches / 12,800 samples.

| Metric | Value |
|---|---:|
| Trained prediction loss | 0.013773 |
| No-change/persistence baseline | 0.014472 |
| Shuffled-action loss | 0.016555 |
| Validation SIGReg | 3.906094 |
| Improvement over persistence | 4.8% |
| Loss increase from wrong actions | 20.2% |

Interpretation: one-step generalization is modest, but the model is measurably
using the actions.

An earlier sequential validation batch produced SIGReg `95.8325`. That number
was invalid because each batch contained heavily overlapping neighboring windows.
Randomizing held-out windows corrected the measurement.

### Held-out five-step rollout

The model received three real context states and then recursively predicted five
future states. The test used 100 batches / 3,200 held-out sequences.

| Metric | Value |
|---|---:|
| Five-step rollout loss | 0.077717 |
| No-change baseline | 0.142473 |
| Shuffled-action rollout loss | 0.114648 |
| Improvement over no-change | 45.5% |
| Loss increase from wrong actions | 47.5% |

This is promising evidence for average short-horizon dynamics, but it does not
prove that the latent distance is useful for planning.

### Offline CEM test

Categorical CEM searches over the 12 legal actions in the Mario Gym's
`COMPLEX_MOVEMENT` action set. Each chosen action is held for five emulator
frames. The planner uses 300 candidates, 30 elites, 10 iterations, and horizon 5.

One held-out sequence produced:

```text
planned actions:
DOWN, A, NOOP, A, NOOP

CEM predicted goal distance:        0.008689
random-sequence predicted mean:     0.026666
recorded-action predicted distance: 0.048007
CEM improvement over random:        67.4%
```

This proves CEM can optimize the model's internal score. The invented sequence
beating the actions that actually produced the goal is also a warning that CEM
may exploit model errors.

### Live NES CEM test

The real NES playground created a simple attainable goal by holding `RIGHT+B`
for 25 emulator frames. The environment was reset to the same initial state,
CEM planned five action blocks, and those actions were executed in the actual
emulator.

```text
planned actions:
RIGHT+B, LEFT+B, UP, RIGHT+B, RIGHT+B

predicted goal distance: 0.092426
predicted random mean:   0.119556

actual goal distance:    0.017372
actual random mean:      0.008830

start x_pos:             40
goal x_pos:              72
CEM x_pos:               44
random mean x_pos:       38.0
terminated:              False
```

**Result: FAIL.** CEM moved Mario slightly farther right than random actions, but
did not approach the goal. More importantly, random final states looked closer
to the goal under latent MSE even though Mario remained near the start. The
current latent Euclidean distance is not a reliable Mario goal metric.

## NES playground

The playable environment uses:

```text
Python 3.13
gym-super-mario-bros 9.1.0
Gymnasium 1.3.0
nes-py 9.0.1
```

It lives in a separate `.venv-mario` so the working Python 3.9 ML environment is
not disturbed.

The headless test completed 120 frames at approximately 2,085 frames/second.

Create and install:

```bash
uv venv --python 3.13 .venv-mario
uv pip install --python .venv-mario/bin/python gym-super-mario-bros==9.1.0
```

Headless test:

```bash
.venv-mario/bin/gym_super_mario_bros \
  --env SuperMarioBros-1-1-v0 \
  --mode random \
  --steps 120 \
  --no-render \
  --seed 42
```

Interactive playground:

```bash
.venv-mario/bin/gym_super_mario_bros \
  --env SuperMarioBros-1-1-v0 \
  --mode human \
  --actionspace complex
```

Keyboard controls:

```text
A / D     left / right
W / S     up / down
O         NES A / jump
P         NES B / run or shoot
Escape    close
```

## Decoder experiments (not part of the active plan)

LeWM does not require a pixel decoder for training or planning. The original
authors trained a separate lightweight decoder only to visualize latent
predictions.

TinyLeWM tested three decoder versions:

1. **Patch-transformer decoder (`decoder-v1`)**
   - Full-data MSE reached roughly `0.11`.
   - Reconstructions had severe stripes and lost almost all objects.
   - Failed the visual gate.
2. **Convolutional decoder (`decoder-v2`)**
   - Eight-image overfit MSE: `0.006320`.
   - Shuffled-latent MSE: `0.836634`.
   - Full-data MSE reached roughly `0.06–0.09`.
   - Reconstructed backgrounds, pipes, ground, HUD, and large geometry, but
     frequently lost Mario and enemies.
3. **Detail-focused fine-tune (`decoder-v3`)**
   - Added hard-pixel and edge losses.
   - Improved visual detail but still did not make Mario reliable enough.

The decoder work is preserved as an experiment, but it has been removed from the
critical path. Gameplay success will be judged using actions executed in the NES
emulator.

## What has and has not been demonstrated

### Demonstrated

- The HDF5 dataset and episode split are valid.
- The JEPA training loop, scheduler, mixed precision, metrics, and checkpoints
  work.
- The model can overfit a tiny fixed sample.
- Prediction loss improves on varied data.
- Held-out predictions use action information.
- Recursive five-step prediction beats a no-change baseline on average.
- CEM can optimize action sequences inside the learned model.
- The real NES environment is installed and controllable from Python.

### Not demonstrated

- Reliable encoding of Mario's position and velocity.
- Reliable enemy, collision, or jump physics.
- A latent goal distance aligned with real Mario progress.
- Successful live goal-conditioned planning.
- Level completion or obstacle avoidance.

## Current diagnosis and next step

The central concern is no longer basic training stability. It is whether the
latent representation preserves control-relevant information and geometry.

Average prediction loss can be dominated by backgrounds, scrolling, platforms,
and other large visual regions. Mario occupies very few pixels. SIGReg prevents
global collapse, but it does not force the representation to prioritize Mario.

### Next diagnostic: frozen position probe

1. Generate several thousand frames in the NES playground.
2. Record each frame with the environment's true `x_pos` and `y_pos`.
3. Encode the frames using the frozen JEPA checkpoint.
4. Train a tiny linear or shallow MLP probe from 192 latent values to `(x, y)`.
5. Evaluate on held-out episodes or trajectories.

Decision:

- **Accurate probe:** Mario is represented, but raw latent MSE is a poor goal
  cost. Use the probe or another reachability-aware cost for CEM.
- **Inaccurate probe:** the encoder does not represent Mario consistently enough.
  More epochs alone may not fix it; consider object-aware data/losses,
  multi-step training, or a different representation.

## Modal storage

Important Volume paths in `tinylewm-storage`:

```text
/data/smb-full.h5
/runs/jepa-one-epoch-v1/jepa_final.pt
/runs/jepa-one-epoch-v1/metrics.jsonl
/runs/live-cem-v1/live_cem_result.png
/runs/live-cem-v1/live_cem_rollout.gif
```

Decoder experiment checkpoints are stored under:

```text
/runs/decoder-v1/
/runs/decoder-v2/
/runs/decoder-v3/
/runs/decoder-overfit-v2/
```

## Reproduction commands

Build and validate the full dataset locally:

```bash
caffeinate -i .venv/bin/python -m datasets.build_hdf5 \
  --src data-smb \
  --out data/smb-full.h5

.venv/bin/python -m tests.validate_smb_dataset \
  --h5 data/smb-full.h5 \
  --batch-size 8

.venv/bin/python -m tests.smb_split \
  --h5 data/smb-full.h5 \
  --val-fraction 0.1 \
  --seed 42
```

Run the main JEPA gates:

```bash
.venv/bin/modal run scripts/modal_tests.py::run_overfit
.venv/bin/modal run scripts/modal_tests.py::run_smoke
.venv/bin/modal run scripts/modal_tests.py::validate_full_data
```

The completed one-epoch training configuration was:

```bash
.venv/bin/modal run --detach scripts/modal_tests.py::train_full_data \
  --max-steps 5125 \
  --batch-size 128 \
  --run-name jepa-one-epoch-v1 \
  --num-workers 8 \
  --log-every 100 \
  --checkpoint-every 1000
```

Held-out evaluation:

```bash
.venv/bin/modal run scripts/modal_tests.py::evaluate_jepa \
  --run-name jepa-one-epoch-v1 \
  --batch-size 128 \
  --max-batches 100 \
  --num-workers 8

.venv/bin/modal run scripts/modal_tests.py::evaluate_rollout \
  --run-name jepa-one-epoch-v1 \
  --batch-size 32 \
  --max-batches 100 \
  --num-workers 8
```

Offline and live CEM:

```bash
.venv/bin/modal run scripts/modal_tests.py::test_cem \
  --run-name jepa-one-epoch-v1 \
  --sample-index 1000

.venv/bin/modal run scripts/modal_tests.py::run_live_cem \
  --run-name jepa-one-epoch-v1 \
  --output-name live-cem-v1
```

## Main files

```text
datasets/build_hdf5.py          build the SMB HDF5
datasets/smb_dataset.py         sliding-window PyTorch dataset
datasets/splits.py              deterministic episode split

models/encoder.py               ViT visual encoder
models/action_encoder.py        action-block encoder
models/predictor.py             autoregressive latent predictor
models/sigreg.py                anti-collapse regularizer
models/jepa.py                  combined JEPA interface
models/decoder.py               inactive visualization experiment

scripts/train_jepa.py           full JEPA training
scripts/overfit_jepa.py         fixed-sample memorization gate
scripts/smoke_jepa.py           short varied-data stability gate
scripts/eval_jepa.py            held-out one-step evaluation
scripts/eval_rollout.py         held-out recursive five-step evaluation
scripts/plan_cem.py             categorical latent-space CEM
scripts/live_cem_mario.py       real NES execution and comparison
scripts/modal_tests.py          Modal entry points

scripts/train_decoder.py        inactive decoder experiment
scripts/render_decoder_samples.py
```

## Known limitations

- Only one JEPA epoch has been trained.
- The dataset contains many long, overlapping windows, so the effective diversity
  is lower than the raw window count suggests.
- Validation covers only 13 of the 32 training levels.
- The HDF5 builder did not preserve the PNG RAM snapshots mentioned in the
  original project plan.
- The current predictor was trained with teacher forcing and only one-step loss;
  recursive predictions can drift out of distribution.
- CEM can exploit model inaccuracies.
- Euclidean latent distance is currently not aligned with Mario's location.
- The first live test covers only a simple 25-frame level 1-1 goal.

## Bottom line

TinyLeWM has passed its data, optimization, checkpoint, one-step, and average
five-step prediction gates. It has **not** passed the live planning gate. The
project is currently a working latent-dynamics research prototype and an honest
negative result for naive Euclidean latent planning on Mario—not yet a working
Mario agent.
