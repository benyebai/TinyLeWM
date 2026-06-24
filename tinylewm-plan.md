# TinyLeWM — Implementation Plan

A from-scratch reimplementation of **LeWorldModel** (Maes et al., March 2026) trained on Super Mario Bros gameplay, in the aesthetic of the TinyWorlds repo. Goal: train a JEPA-style latent world model on ~700k SMB frames, then demo latent planning with a split-screen "imagined vs. actual" rollout.

---

## 0. Project Identity

- **Name:** TinyLeWM
- **Inspiration:** LeWM paper (2603.19312v2) + TinyWorlds repo structure
- **Data:** `data-smb/` — 280 episodes, ~737k frames, 9 players, 32 levels, 26 action codes
- **Compute:** Single H100, ~6 hours total training
- **Lines of code (target):** 1,500–2,500
- **Demo:** Option C — split-screen showing the model's imagined latent rollout decoded to pixels alongside the actual emulator rollout, under the same action sequence chosen by CEM planning.

---

## 0.5. Dataset Adequacy (verified against App. E)

| Dataset | Episodes | Avg length | Total frames | Epochs |
|---|---|---|---|---|
| TwoRoom | 10,000 | 92 | ~920k | 10 |
| PushT | 20,000 | 196 | ~3.92M | 10 |
| OGBench-Cube | 10,000 | 200 | ~2.0M | 10 |
| Reacher | 10,000 | 200 | ~2.0M | 10 |
| **TinyLeWM** | **280** | **~2,632** | **~737k** | **10** |

- ~737k frames is ~80% of TwoRoom (their smallest). Workable.
- Episodes are 13–28× longer than the paper's, so we have more temporal diversity per trajectory.
- With sliding-window sub-trajectory sampling at frame-skip 5, unique sub-trajectory count (~731k) matches TwoRoom (~720k).
- Caveat: Mario is visually much richer than TwoRoom (sprites vs. two dots). Same data count, harder representation problem. Plan for slightly weaker rollout quality than the paper's TwoRoom numbers.

## 1. What LeWM Actually Is

Two losses, end-to-end, no decoder during training:

1. **Predictive loss** — encode frame `x_t` → embedding `s_t`. A transformer predictor takes `(s_{t-H..t}, a_{t..t+K-1})` and outputs `ŝ_{t+K}`. Train with MSE against `s_{t+K}`, **with gradients flowing through both sides** (the prediction target is just the same encoder running on a future frame in the same batch — no EMA, no stop-gradient, no target copy).
2. **SIGReg** — Sketched-Isotropic-Gaussian Regularizer. Project embeddings onto random unit vectors, compare the resulting 1D distributions to N(0,1) using the Epps–Pulley statistic. Justified by the Cramér–Wold theorem (a distribution is determined by its 1D projections).

That's it. No contrastive loss. No reconstruction. No latent actions. No EMA or stop-gradient (LeWM's central pitch is that SIGReg alone prevents representation collapse — see Section 3.1 of the paper: "We do not employ stop-gradient, exponential moving averages, or additional stabilization heuristics"). The representation is **shaped** by SIGReg and **made predictive** by the MSE loss, with the whole system trained jointly end-to-end.

For planning: encode a goal frame `s_g`, run CEM over action sequences in latent space, pick the sequence whose final predicted embedding minimizes `||ŝ_{t+H} − s_g||²`, execute first action, replan (MPC).

---

## 2. Why Mario Works for This

- **Visually simple** — 2D sprites, deterministic transitions, finite levels
- **Plenty of data** — 700k frames is more than the paper's PushT setup
- **Discrete actions** — 6-bit multi-hot (A, B, Up, Down, Left, Right); drop Select/Start
- **Short demo horizon** — H=5 latent steps × frame-skip 5 = 25 emulator frames ≈ 0.4s of gameplay
- **Risk:** Mario is harder than PushT because the goal-state encoding has to be discriminative about Mario's pixel position. Plan accordingly (see §7 risks).

---

## 3. Reading Checklist (done)

- [x] LeWM paper §3.1 (training), §3.2 (planning), App. A (SIGReg), App. B (CEM), App. D (impl details)
- [x] Genie paper (Module overview only, for contrast with TinyWorlds)
- [x] TinyWorlds repo structure
- [x] DIAMOND/Dreamer high-level (for "what other world models look like")

---

## 4. Repo Structure (TinyWorlds-flavored)

```
tinylewm/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── tinylewm.yaml          # all hyperparameters
├── datasets/
│   ├── build_hdf5.py          # PNG → HDF5 conversion
│   └── smb_dataset.py         # PyTorch Dataset over HDF5
├── models/
│   ├── encoder.py             # ViT-Tiny + BN-MLP projector
│   ├── predictor.py           # 6-layer transformer + AdaLN-Zero
│   ├── sigreg.py              # Epps–Pulley regularizer
│   └── decoder.py             # frozen-encoder pixel decoder (Phase 4)
├── scripts/
│   ├── train_jepa.py          # main training loop
│   ├── train_decoder.py       # post-hoc decoder training
│   ├── eval_rollout.py        # k-step MSE diagnostics
│   ├── plan_cem.py            # CEM planning + MPC
│   └── make_demo.py           # split-screen video generation
├── utils/
│   ├── action_codes.py        # int code → 6-bit multi-hot
│   └── viz.py                 # frame grid, PCA scatter
└── assets/
    └── demo_clips/            # final mp4 outputs
```

---

## 5. Phase-by-Phase Plan

### Phase 1 — Data pipeline (Steps 1–4)

**Confirmed dataset facts** (from the SMB dataset README, [Pinto 2021](https://github.com/rafaelcp/smbdataset)):

- 737,134 frames across 280 episodes (141 wins + 139 fails), 32 levels, recorded by 1 player across multiple sessions
- 256×240 8-bit indexed-color PNGs (mode P — requires `convert("RGB")` on decode)
- Action stored as integer 0–255 in filename `_aXX_`; each bit = one NES button (MSB → LSB: A, Up, Left, B, Start, Right, Down, Select)

| Bit | Value | Button | In our data? |
|---|---|---|---|
| 7 | 128 | A (jump) | yes |
| 6 | 64 | Up (climb) | yes |
| 5 | 32 | Left | yes |
| 4 | 16 | B (run/fire) | yes |
| 3 | 8 | Start | never set |
| 2 | 4 | Right | yes |
| 1 | 2 | Down (pipe) | yes |
| 0 | 1 | Select | never set |

The 6-dim multi-hot drops Select and Start. **Slot order: `[Left, Right, Up, Down, A, B]`** corresponding to source bits `[5, 2, 6, 1, 7, 4]`. PNGs also carry a `tEXtRAM` chunk (2048-byte NES RAM snapshot) — preserved in metadata for Phase 6 probing, unused for training.

**Step 1. Repo skeleton.** Create directories from Section 4. Stub each `.py` with a header docstring. Add empty `__init__.py` in `utils/` and `datasets/` to make them importable packages. Write `requirements.txt` (Phase 1 minimum: `torch`, `numpy`, `pillow`, `tqdm`, `matplotlib`, `h5py`, `pandas`; later phases add `einops`, `opencv-python`, `gym-super-mario-bros`, `nes-py`, `hydra-core`, `wandb`, `imageio`). `.gitignore` excludes `*.h5`, `data-smb/`, `__pycache__/`, `wandb/`, `assets/*.png`. `README.md` stub. `git init`.

**Step 2. Action encoder.** `utils/action_codes.py`. Implement `action_code_to_multihot(code: int) → np.ndarray[6, uint8]` via bit-shift extraction over the confirmed `BUTTON_BITS` table above. Add vectorized `action_codes_to_multihot(codes: np.ndarray) → np.ndarray[N, 6]` for batch use in the HDF5 builder. Unit-test against:

- `a0` → `[0, 0, 0, 0, 0, 0]` (idle)
- `a4` → `[0, 1, 0, 0, 0, 0]` (Right only)
- `a20` → `[0, 1, 0, 0, 0, 1]` (Right + B = run right, most common action)
- `a148` → `[0, 1, 0, 0, 1, 1]` (Right + B + A = running jump)
- `a48` → `[1, 0, 0, 0, 0, 1]` (Left + B = run left)

**Step 3. HDF5 builder.** `datasets/build_hdf5.py`. Walks `data-smb/`, parses each filename into (player_id, sessid, episode, level, frame_idx, action_code, outcome), sorts within each episode by `frame_idx`, decodes PNGs via `PIL.Image.open(...).convert("RGB")` to uint8 `[240, 256, 3]` (PIL gives H×W×C → 240 height × 256 width). Writes one file `data/smb.h5`:

- `/frames` uint8 `[737134, 240, 256, 3]`, gzip level 4, `chunks=(64, 240, 256, 3)` (~12 MB/chunk; 4 strided frames hit one chunk → one disk read per sub-trajectory)
- `/actions` uint8 `[737134, 6]`
- `/frame_metadata` structured `[737134]` of (player_id S32, sessid S32, episode i4, level S8, frame_idx i4, outcome S8)
- `/episodes` int64 `[280, 2]` of (start_idx, length)

Expected compressed size ~17 GB (raw 50 GB; gzip 4 yields ~3× on game frames). Build time ~30 min single-threaded with `tqdm`. **Spot-check mid-build**: print frame 100 and 101 of episode 0, decode their action codes, confirm pixel motion matches the action multi-hot.

**Step 4. PyTorch Dataset + sanity check.** `datasets/smb_dataset.py`. `SMBSubTrajectoryDataset` yields sub-trajectories matching paper App. D: **4 frames at indices [t, t+5, t+10, t+15] + 4 action-blocks of 5 actions each** (frame-skip 5). Bilinear resize 240×256 → 224×224 via `cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)`. Normalize uint8 → float32 in [−1, 1] via `(x/127.5) − 1`. Transpose NHWC → NCHW so frames are `[4, 3, 224, 224]`. Valid starting points filtered so no sub-trajectory straddles an episode boundary. **Use lazy-open HDF5 pattern** (open file in `_get_h5()` on first `__getitem__` call per worker) because `h5py.File` is not fork-safe across DataLoader workers — this is the single most common HDF5+PyTorch bug.

**Action-frame alignment (gym/nes-py convention).** The action stored in filename `f100_a20` is the controller state held WHILE frame 100 is rendered — it's the action causally producing the transition `f_100 → f_101`. So `action_block[i]` of a sub-trajectory starting at frame `s` is `actions[s+5i : s+5i+5]` (e.g. block 0 = actions at indices `[s, s+1, s+2, s+3, s+4]`, which transition `f_s → f_{s+5}`). The 4th action block (`[s+15..s+19]`) has no observed target inside the sub-trajectory window — its prediction is dropped by the MSE loss slicing (`emb[:, 1:]` vs `next_emb[:, :-1]`). **Spot-check during Step 3**: find two consecutive frames where Mario clearly moves right, verify the Right bit is set in the BEFORE frame's action (not the after frame's). 95% confidence this is the convention, 100% after the spot-check.

**Verification gate**: `scripts/phase1_sanity.py` loads one batch and asserts:

- `batch["frames"]` shape `[128, 4, 3, 224, 224]`, dtype float32, value range in [−1, 1]
- `batch["actions"]` shape `[128, 4, 5, 6]`, dtype float32, values in {0, 1}

Save a 1×4 grid of one sub-trajectory's frames with multi-hot action labels overlaid as `assets/phase1_sanity.png`. Eyeball: frames look like Mario, are in temporal order, action labels align with visible motion between frames. **This image is the proof Phase 1 works and the gate to Phase 2.**

### Phase 2 — Model code (Steps 5–9)

**Step 5. ViT-Tiny encoder + projector.** `models/encoder.py`. ViT-Tiny: patch size 14, hidden 192, depth 12, heads 3, ~5M params (use HuggingFace `transformers` or `timm` `vit_tiny_patch16_224` and swap patch size). Take the `[CLS]` token of the last layer as the raw image embedding (192d). Then **one** projector: 1-layer MLP with BatchNorm — `Linear(192 → 192) → BN(192)`. Paper note: this projector exists specifically to break the final LayerNorm of the ViT, which otherwise prevents SIGReg from shaping the embedding distribution. **Single network, no EMA, no target copy** — the file is just one `nn.Module` that does encode-then-project.

**Step 6. Predictor + projector.** `models/predictor.py` — **ALREADY WRITTEN**, see `predictor.py` in the workspace root. Transformer with **6 layers, 16 attention heads, hidden 384, 10% dropout** (per Section 3.1; with full AdaLN-Zero this comes to ~16M params, not the paper's ~10M — the discrepancy is because LeWM likely uses simpler AdaLN without the gate parameters; we use full AdaLN-Zero for stability/identity-at-init which is the more standard variant). Input embedding `Linear(192 → 384)` projects encoder outputs into the predictor's working dim. Action embedding `Linear(30 → 384)` where 30 = 5 actions × 6 buttons flattened per timestep. Learned positional embeddings. Causal mask over history (configurable; default `max_history=8` to allow rollout). **AdaLN-Zero action conditioning**: each block has 6 modulation tensors produced from the action embedding (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp). The modulation Linear is zero-initialized so all gates start at 0 → each block is identity at init → predictor starts as pass-through, learns to use actions gradually. Final `Linear(384 → 192)` maps back to embedding space, followed by a second projector with the same architecture as the encoder's: `Linear(192 → 192) → BN(192)`. Output: `ẑ_{t+1} ∈ ℝ^192`. Used autoregressively at planning time.

**Step 7. SIGReg loss.** `models/sigreg.py`.
- Sample `M=1024` random unit vectors `u_m ∈ ℝ^192` each step (re-sample is fine).
- For each `u_m`: project the batch of embeddings → 1D values `h_m = Z u_m`.
- Compute Epps–Pulley statistic via numerical integration on `t ∈ [0.2, 4]` with `K=32` Gauss–Legendre nodes.
- Average over `m`. Then average across timesteps (step-wise SIGReg from the paper's pseudocode: `mean(SIGReg(emb.transpose(0, 1)))`).
- Multiply final value by `λ=0.1`.

**Step 8. Wire up loss.** Forward pass per the paper's Algorithm 1 pseudocode:
```python
emb = encoder(obs)               # (B, T, D=192)
next_emb = predictor(emb, actions)  # (B, T, D=192), teacher-forced
pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])
sigreg_loss = SIGReg(emb.transpose(0, 1)).mean()   # step-wise
loss = pred_loss + 0.1 * sigreg_loss
```
**Both sides of the MSE see gradient.** No `detach()`, no stop-gradient, no EMA target. Sanity check: SIGReg should drop toward zero when fed a batch of unit-variance Gaussian noise.

**Step 9. Training script.** `scripts/train_jepa.py`. AdamW (lr=3e-4, wd=0.05), cosine schedule with 1000-step warmup, batch 128, bf16 autocast, gradient clip 1.0, 10 epochs (~30k steps). All parameters (encoder, both projectors, predictor) optimized jointly in one pass. Wandb logging: `pred_loss`, `sigreg_loss`, total loss, lr, grad norm, embedding covariance rank (collapse early-warning). Checkpoint every 5k steps. **No EMA update step — there's no target network to update.**

### Phase 3 — Train + diagnose (Steps 10–14)

**Step 10. Smoke test.** 200 steps on 1k frames, verify loss goes down and no NaN.

**Step 11. Full train.** ~6 hours on H100. Watch MSE plateau, SIGReg stay bounded.

**Step 12. Embedding diagnostics.** Hold out one episode. PCA scatter of `s_t` over a level — should show clean trajectory, not collapse. Verify rank of embedding covariance > 50 (no representational collapse).

**Step 13. 1-step rollout.** Encode `x_t`, predict `ŝ_{t+1}`, compute `||ŝ_{t+1} − s_{t+1}||²`. Compare to "predict the current embedding" baseline. Must beat it cleanly.

**Step 14. 5-step rollout (DECISION GATE).** Same as Step 13 but K=5. Compare to do-nothing baseline.
- **≥30% better** → green light Phase 4
- **10–30% better** → cautious yellow; try lr or λ tweaks
- **≈baseline** → red; debug encoder/predictor before continuing

### Phase 4 — Decoder for visualization (Steps 15–17)

The decoder is **only for the demo video**. It does not affect representation quality.

**Step 15. Decoder arch.** Cross-attention transformer decoder (matches paper App. D).
- Project the encoder's [CLS] token (192d) → hidden dim; use as key/value for cross-attention.
- 196 learnable query tokens (one per 16×16 patch of the 224×224 image — patch size **16**, different from encoder's patch 14).
- Several cross-attention layers with residual MLP blocks.
- Linear projection of each query output → 16×16×3 pixel patch; rearrange to 224×224 RGB.

**Step 16. Train decoder.** Freeze the encoder. Train decoder to reconstruct via L2 + LPIPS loss. ~1 hour on H100. This is a diagnostic visualization tool only — it does not affect representation quality.

**Step 17. Sanity check.** Encode → decode known frames; verify they look like Mario (blurry is OK, recognizable is required).

### Phase 5 — CEM planning (Steps 18–20)

**Step 18. Categorical CEM.** `scripts/plan_cem.py`. Matches paper App. D (non-PushT settings):
- Maintain per-step Bernoulli probabilities for each of 6 buttons (init at 0.5, init variance 1.0).
- Sample **N=300** candidate action sequences of length **H=5**.
- Roll out in latent space: `s_t → ŝ_{t+1} → ... → ŝ_{t+H}` (autoregressive predictor).
- Score by `||ŝ_{t+H} − s_g||²`.
- Take **top 30** (top 10%) elites, refit Bernoulli probs per step.
- Iterate **10 times**.

**Step 19. MPC.** Execute the **entire optimized 5-action sequence** before replanning (receding-horizon, matches paper App. D). Re-observe, re-encode, replan from the new state.

**Step 20. Goal selection.** Three demo goals, easiest first:
- (a) Local navigation: `s_g` = encode of a frame 25 emulator frames ahead in same episode (high-confidence win)
- (b) Obstacle: jump over a Goomba (medium)
- (c) Reach end of level segment (stretch, may not work)

### Phase 6 — Demo video (Steps 21–22)

**Step 21. Split-screen renderer.** `scripts/make_demo.py`.
- Left pane: actual emulator rollout under CEM-selected actions.
- Right pane: decoder-rendered imagined rollout from the same actions.
- Overlay: chosen action bitmap, planning step counter, "imagined" vs. "actual" labels.

**Step 22. Polish.** Pick 3–5 best clips. Add intro card explaining JEPA / SIGReg in one slide. Export 720p mp4.

### Phase 7 — Writeup (Step 23, optional)

Single-page report: what LeWM is, what changes were made for Mario (categorical CEM, frame-skip 5, 6-bit actions), diagnostic curves, demo gif.

---

## 6. Hyperparameter Summary

| Knob | Value | Source |
|---|---|---|
| Image size | 224 × 224 (center crop) | TinyLeWM choice |
| Frame skip | 5 | Matches NES @ 60Hz → 12 Hz effective |
| History length H | 3 frames | LeWM paper |
| Prediction horizon K | 5 actions | LeWM paper |
| Encoder | ViT-Tiny (192d, 12L, 3H, patch 14, ~5M params) + 1-layer Linear(192→192)+BN projector | LeWM paper §3.1 |
| Predictor | 6L, 16H, hidden 384, 10% dropout, ~10M params, AdaLN-Zero action conditioning + 1-layer Linear(192→192)+BN projector | LeWM paper §3.1 |
| SIGReg M | 1024 projections | LeWM paper App. A |
| SIGReg K | 32 Gauss–Legendre nodes | LeWM paper App. A |
| SIGReg integration | [0.2, 4] | LeWM paper App. A |
| λ (SIGReg weight) | 0.1 | LeWM paper |
| Optimizer | AdamW, lr 3e-4, wd 0.05 | LeWM paper |
| Schedule | Cosine, 1000-step warmup | LeWM paper |
| Batch size | 128 | LeWM paper App. D |
| Total steps | ~30k (10 epochs) | LeWM paper App. E |
| Action encoding | 6-bit multi-hot (A,B,U,D,L,R) | TinyLeWM choice |
| Planning CEM N | **300 candidates, 10 iters, top 30**, init var 1.0 | LeWM paper App. D |
| MPC scheme | Execute full 5-action sequence, then replan | LeWM paper App. D |

---

## 7. Risk Assessment — Will Planning Work?

Honest probabilities for the three demo goals:

- **Local navigation (25-frame target):** 70–80% — short horizon, target is in-distribution, no compounding errors
- **Obstacle avoidance (one Goomba):** 30–50% — needs precise positional encoding
- **Level completion (long horizon):** 10–20% — compounded prediction errors will dominate

**Mitigations baked into the plan:**
- Decision gate at Step 14 — if 5-step rollout doesn't beat baseline by 30%+, don't waste a day building the decoder
- Three goals of escalating difficulty — even if (c) fails, (a) makes a working demo
- Split-screen format makes "imagined diverges from actual" visually interesting rather than embarrassing

**Known unknowns:**
- Goal-state matching in latent space — encoder isn't trained to make pixel-distance map to embedding-distance
- Whether SIGReg's "isotropic Gaussian" target representation is compatible with Mario's sparse action manifold
- Whether 6 buttons × 5 steps = 30-bit action search is small enough for CEM (it is: 2^30 ≈ 10⁹ but CEM never enumerates)

---

## 8. De-risking Ladder — What to Run Before Full Training

**Do not write everything then hit "go" on a 6-hour training run.** Five cheap validation levels, each catching a different class of bug. Skipping any of them is asking for pain.

### Level 1 — Phase 1 sanity image (laptop, ~30 min)

```bash
python datasets/build_hdf5.py        # ~30 min single-threaded
python scripts/phase1_sanity.py
```

Pass if: HDF5 file ~17 GB, `assets/phase1_sanity.png` shows recognizable Mario in temporal order with sensible action labels, `batch["frames"].shape == (128, 4, 3, 224, 224)`, `batch["actions"].shape == (128, 4, 5, 6)`, values in [−1, 1].

Fail modes: scrambled colors (PNG mode P conversion missing), axis flip (H/W swapped), all-zero actions (filename parser or encoder bug), straddling episodes (`valid_starts` filter wrong).

### Level 2 — Each model file's `__main__` (laptop, ~2 min)

```bash
python models/encoder.py
python models/predictor.py
python models/sigreg.py
```

Pass if:
- Encoder: ~5.7M params, forward on `[B, 3, 224, 224]` returns `[B, 192]`
- Predictor: ~16M params (with full AdaLN-Zero; ~10M if dropping gates), forward on `[B, T, 192]`+`[B, T, 30]` returns `[B, T, 192]`
- SIGReg: small value on `torch.randn(128, 192)`, LARGE value on `torch.zeros(128, 192)` (must distinguish Gaussian from collapsed)

### Level 3 — Overfit on 10 samples (any 16GB+ GPU, ~10 min)

**THE single most important test.** If the model can't memorize 10 sub-trajectories, it can't learn anything.

```python
# scripts/test_overfit.py
small_ds = torch.utils.data.Subset(ds, list(range(10)))
# Train 2000 steps on the same 10 samples
# Expect pred_loss to drop from ~1.0 to <0.01
```

Pass if: `pred_loss` drops to near-zero within 2000 steps.

Fail modes:
- Loss stays at ~1.0 → gradients aren't reaching the encoder or predictor (check `.grad` after `.backward()`)
- Loss drops to 0 instantly → encoder collapsed to constant; SIGReg isn't doing its job
- Loss NaN → numerical instability; lower lr or enable grad clip
- Loss oscillates wildly → lr too high

### Level 4 — 200-step smoke test (any 16GB+ GPU, ~20 min)

```bash
python scripts/train_jepa.py --max_steps 200 --log_every 10
```

Pass if: loss curves trend down without NaN, grad norm in 0.1–10 range, embedding covariance rank stays > 100, steps/sec in expected range for your GPU.

### Level 5 — Full training (real GPU, ~6 hours)

Only run after Levels 1–4 all pass. Checkpoint every 5k steps for rollback safety.

### GPU Requirements

| GPU | VRAM | Full training time | Use case |
|---|---|---|---|
| H100 | 80 GB | ~3 hr | Best |
| L40S (paper's choice) | 48 GB | ~6 hr | Paper-matching |
| A100 40GB | 40 GB | ~6 hr | Common cloud option |
| RTX 4090 | 24 GB | ~8–10 hr | Workstation |
| RTX 3090 | 24 GB | ~12 hr | Workable |
| A10G (Databricks `g5.xlarge`) | 24 GB | ~10 hr | **Sweet spot for cost** |
| Colab T4 free | 16 GB | 20+ hr | Sanity checks only |

Use bf16 autocast (free 2× on Ampere+), `torch.compile` if available (~1.3× more), gradient accumulation if batch 128 doesn't fit.

---

## 9. Timeline (rough)

| Day | Phase | Hours |
|---|---|---|
| 1 | Phase 1 (data) + Levels 1 | 4–6 |
| 2 | Phase 2 (model code) + Level 2 | 6–8 |
| 3 | Levels 3–4 (overfit + smoke) | 3 |
| 4 | Phase 3 — Level 5 (full train) + diagnostics | 6 train + 2 analyze |
| 5 | Phase 4 (decoder) | 4 |
| 6 | Phase 5 (CEM planning) | 4–6 |
| 7 | Phase 6 (demo) + Phase 7 (writeup) | 4 |

**Total:** ~7 working days, ~35 hours of focused work.

---

## 10. Current Progress

**Completed:**
- [x] Read paper Sections 3.1, 3.2, Appendices A, D, E
- [x] Verified dataset adequacy (737k frames vs paper's TwoRoom 920k baseline)
- [x] Confirmed action bit-to-button mapping from SMB dataset README (`A=128, Up=64, Left=32, B=16, Right=4, Down=2`; Select/Start never set)
- [x] Wrote `predictor.py` (in workspace root — needs to be copied to `tinylewm/models/predictor.py`)
- [x] Phase 1 Step 1 (repo skeleton)

**In progress:**
- [ ] Phase 1 Step 2 — `utils/action_codes.py` (action encoder)

**Next up (in order):**
1. Finish `utils/action_codes.py` (Step 2)
2. Write `datasets/build_hdf5.py` (Step 3) — biggest data step
3. Write `datasets/smb_dataset.py` (Step 4)
4. Run Level 1 (Phase 1 sanity)
5. Write `models/encoder.py` (Step 5, ~20 lines wrapping HF ViTModel)
6. Move existing `predictor.py` into `tinylewm/models/predictor.py`
7. Write `models/sigreg.py` (Step 7)
8. Run Level 2 (model file self-tests)
9. Write `scripts/train_jepa.py` and `scripts/test_overfit.py`
10. Run Levels 3, 4, then 5

**Files already written (in this session):**
- `predictor.py` at workspace root — copy to `tinylewm/models/predictor.py`

**Files NOT yet written (to do):**
- `utils/action_codes.py`
- `datasets/build_hdf5.py`
- `datasets/smb_dataset.py`
- `scripts/phase1_sanity.py`
- `models/encoder.py`
- `models/sigreg.py`
- `models/decoder.py` (Phase 4 only)
- `scripts/test_overfit.py`
- `scripts/train_jepa.py`
- `scripts/eval_rollout.py`
- `scripts/plan_cem.py`
- `scripts/make_demo.py`

---

## 11. Key Decisions & Conventions (one-pagers for handoff)

### Action encoding
- Source: SMB dataset README ([github.com/rafaelcp/smbdataset](https://github.com/rafaelcp/smbdataset))
- Bit-to-button: `bit 7=A, 6=Up, 5=Left, 4=B, 3=Start, 2=Right, 1=Down, 0=Select`
- 6-dim multi-hot order: `[Left(bit5), Right(bit2), Up(bit6), Down(bit1), A(bit7), B(bit4)]`
- Bits 0 (Select) and 3 (Start) are never set in our data — dropped
- Action stored at frame N is held DURING frame N, applied to produce frame N+1
- Action block for transition `f_s → f_{s+5}` = `actions[s : s+5]`

### Data shapes through the pipeline
```
DataLoader output:      [128, 4, 3, 224, 224]    frames + [128, 4, 5, 6]    actions
                                                    ↓ flatten time
Encoder input:          [512, 3, 224, 224]
Encoder output:         [512, 192]
                                                    ↓ reshape
Predictor input:        emb [128, 4, 192]  +  actions [128, 4, 30]    (5×6 flattened)
Predictor output:       next_emb [128, 4, 192]
Loss:                   F.mse_loss(emb[:, 1:], next_emb[:, :-1])
SIGReg:                 mean(SIGReg(emb.transpose(0, 1)))   # step-wise
Total loss:             pred_loss + 0.1 * sigreg_loss
```

### Training principles (paper-faithful)
- **No EMA target encoder.** Single encoder, both sides of MSE see gradient.
- **No stop-gradient.** End-to-end backprop through everything.
- **No pretrained weights.** ViT-Tiny initialized random, trained from scratch.
- **Two losses, joint optimization.** AdamW updates encoder + predictor + projectors + AdaLN params all together.
- **The projector exists** specifically to break the encoder's final LayerNorm so SIGReg can shape the embedding distribution.

### Hardware
- Target: any 24GB+ GPU. Sweet spot for cost: Databricks `g5.xlarge` (A10G, 24 GB).
- bf16 autocast + `torch.compile` recommended.

---

## 12. Next Action

**Continue Phase 1 Step 2** — write `utils/action_codes.py` with the confirmed `BUTTON_BITS` mapping and unit tests against the top-5 action codes (a0, a4, a20, a148, a48). Then move on to Step 3 (HDF5 builder).
