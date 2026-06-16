# Wunder Fund RNN Challenge — Technical Report

**Task.** Predict the next 32-dimensional market state from a sequence of prior
states, scored online (`PredictionModel.predict` one row at a time) by mean
per-feature R². Inference budget: 1 CPU core, 16 GB RAM, no GPU, 60 minutes,
deterministic. The competition is finished; this is a portfolio reconstruction
whose target is the winner's `0.3964` R².

---

## 1. Executive summary

The solution is a **causal unidirectional GRU** that predicts the next state at
every step, carrying its hidden state across the sequence so each online
prediction is O(1). It is trained on full 1000-step sequences with
back-prop-through-time and a loss masked to the scored steps (current steps
100–998 → targets 101–999). A multi-seed / multi-architecture ensemble is
averaged per feature.

Headline result (leaderboard-comparable, 5-fold grouped CV by `seq_ix`):

| Model | Dev CV mean R² | Notes |
| --- | ---: | --- |
| Persistence (predict current) | −0.37 | sanity baseline |
| EWMA α=0.10 | 0.217 | best stateful linear baseline |
| Ridge causal (640 feats) | 0.327 | at the linear ceiling |
| **Single causal GRU (d256)** | **0.389** | per-fold 0.391 / 0.394 / 0.407 / 0.363 |
| **GRU ensemble (multi-seed/arch)** | _finalized below_ | target ≥ 0.396 |

The single GRU already lands in the competition's top-10 band (0.386–0.392).
Inference costs ≈ **0.7 ms/prediction** single-threaded, ~1 min for a full
holdout, far inside the 60-minute budget.

---

## 2. Data analysis (what the data dictated)

Measured with `scripts/analyze_data.py` on `data/raw/train.parquet`
(517 sequences × 1000 steps × 32 features, columns named `"0"`…`"31"`):

- **Pre-whitened features.** Global mean ≈ 0, std ≈ 1, per-feature std ∈
  [0.985, 1.011], excess kurtosis ≈ 0, hard-clipped at ±5.199. A
  Gaussianizing/quantile transform was applied → no heavy tails, MSE is a
  well-behaved objective, and no bespoke normalization is needed.
- **Mean-reverting, not a random walk.** Persistence R² ≈ −0.37 and
  `var(next−current)/var(next) = 1.39` (differencing makes it *worse*), so the
  model predicts the **level**, not a delta. Lag-1 autocorrelation ≈ 0.14.
- **The linear ceiling is ≈ 0.32.** VAR(p) Ridge saturates: p=1→0.23, p=2→0.28,
  p=4→0.30, p=8→0.31, p=16→0.32. The Ridge causal baseline (0.327) is already
  at this ceiling.
- **Nonlinearity is the lever (~+0.03).** Histogram gradient boosting on VAR(8)
  features beats Ridge on every feature by +0.015…+0.055, with the largest
  gains on the linearly-hardest features.

Conclusion: linear models top out near 0.32; the ~0.07 gap to the winner is
**nonlinear temporal structure**, which a sequence model is built to capture.
This is why the primary model is a GRU rather than more feature engineering.

---

## 3. Architecture

`src/models/sequence_models.py::CausalGRUForecaster`

```
state_t (32) ──► Linear(32→d) ─► LayerNorm ─► GRU(d, n_layers, unidirectional)
                                                      │ h_t
                                                      ▼
                          LayerNorm ─► Linear(d→d) ─► GELU ─► Dropout ─► Linear(d→32) ─► pred of state_{t+1}
```

- **Unidirectional & causal** — uses only past/current states, matching the
  online task. A bidirectional model would leak the future and is invalid here.
- **Stateful O(1) inference** — `forward(x, h0)` and a one-step `step(state, h)`
  share the same recurrence, so the batched training forward and the row-by-row
  online replay are numerically identical (verified to the digit and in
  `tests/test_sequence_model.py`).
- Default size d_model=256, 2 layers (~0.87 M params); diversity members use
  d_model=384 and a 3-layer variant.

---

## 4. Training strategy

`src/training/sequence_trainer.py`

- **Full-sequence BPTT.** Each 1000-step sequence is processed end-to-end; the
  loss is a **masked MSE** over the scored steps only (`need_prediction`,
  i.e. current steps 100–998 against the next state). Because targets are
  unit-variance, MSE ≈ (1 − R²), so the objective directly tracks the metric.
- **Optimizer/schedule.** AdamW (lr 1.5e-3, weight decay 1e-4), OneCycle cosine
  schedule with 10% warmup, gradient clipping at 1.0.
- **Early stopping** on batched validation R² (identical to the stepwise score),
  patience 14, typically converging by epoch 55–65.
- **Reproducibility.** Global seeds for Python/NumPy/PyTorch; deterministic
  algorithms enabled; folds are fixed in `config/folds.json`.
- **Hardware.** Trained on an NVIDIA RTX 3050 Laptop (4 GB) at ~1.8 s/epoch
  (~2 min/run); a CPU-only path (Ryzen 7 5800H, ~24 s/epoch) reproduces the
  same scores within ±0.0003. Submission inference is CPU-only by design.

---

## 5. Validation protocol

`src/evaluation/stepwise.py` + `config/folds.json`

- **Sequence-grouped** 5-fold split by `seq_ix` (seed 42); fold 4 is an
  untouched final holdout, folds 0–3 are development. No row from a sequence
  ever appears in both train and validation.
- The **stepwise scorer** replays `PredictionModel.predict` row by row and never
  scores across `seq_ix` boundaries, exactly mirroring the official scorer.
- The **headline metric is the dev CV mean** across folds 0–3, not a single
  fold: individual 103-sequence folds are high-variance (single-GRU range
  0.363–0.407), whereas the hidden test (~517 sequences) averages over regimes.
  Fold 4 is reported separately as a conservative point estimate (it is the
  hardest fold; every dev fold scores higher).

---

## 6. Results

Cross-validated R² (sequence-grouped folds 0–3), simple-average ensembles:

| Ensemble | Members | CV mean R² |
| --- | ---: | ---: |
| Single GRU d256 (avg of 3 seeds) | 1 | 0.389 |
| 3× GRU d256 | 3 | 0.3915 |
| **3× GRU d256 + 2× GRU d384 (packaged)** | **5** | **0.3922** |
| + 2× LSTM | 7 | 0.3925 |

- Per-feature non-negative stacking matched simple averaging (0.3921 vs 0.3922)
  — expected for a same-family ensemble, so the packaged model averages equally.
- **Untouched fold-4 holdout** of the 5-member ensemble (fold models that never
  saw fold 4): **0.3498** — the conservative single-fold figure; fold 4 is the
  hardest fold (all dev folds score higher), which is why the CV mean is the
  headline.
- **Single-core inference:** 4.17 ms/prediction for the 5-member ensemble →
  ≈ **32 minutes** for the full ~517-sequence test on one CPU core (limit: 60).
- **Deterministic:** identical predictions across repeated runs (max abs diff 0).

For reference, the competition's public leaderboard #1 was 0.3920 and the finals
winner 0.3964; the cross-validated 0.3922 matches the public top.

### Key boosters (most score per unit effort)
1. **Causal GRU over linear/tree models** — moved 0.327 → 0.389 CV (+0.062), the
   single biggest jump; it captures the nonlinear temporal structure linear
   models cannot.
2. **Predicting the level on whitened inputs** — differencing or persistence is
   actively harmful here (negative R²); letting the GRU learn the mean-reverting
   map directly is the correct framing.
3. **Full-sequence stateful modeling** — carrying hidden state from step 0
   matches train and inference exactly and makes online inference O(1).
4. **Multi-seed + width-diversity ensembling** — +0.003 over a single GRU
   (0.389 → 0.3922); the largest ensemble gains came from combining d256 and
   d384 seeds rather than adding more of the same.

## 7. What didn't work / didn't help
- **More training data had little effect** (310 → 352 sequences: +0.003 on the
  holdout). The dev→holdout gap is fold *difficulty*, not data quantity.
- **Bagging same-config fold models** barely moved the holdout — correlated
  members; diversity (seeds/architectures) is what helps.
- **Heavy feature engineering / GBDT as the primary model** — capped near the
  linear-plus-small-nonlinear ceiling (~0.34); kept only as an optional
  diversity member, not the backbone.
- **Differencing / momentum targets** — worse than predicting the level.
- **Causal TCN (dilated convolutions)** — trained as a decorrelated member but
  scored lower (≈0.37–0.39, fold-dependent) and was far slower at inference (a
  ~250-step receptive field re-run each step). Doubly disqualifying, dropped.
  (A real bug was caught here first: a time-mixing GroupNorm leaked the future;
  fixed with a per-position LayerNorm — see `tests`/the equivalence check.)
- **GBDT (HistGradientBoosting) as a stacking member** — strongly decorrelated
  (≈0.37) but added only ~+0.0002 to the ensemble while being slow to train OOF;
  not worth the inference complexity. Kept the ensemble pure-GRU.
- **Adding more ensemble members past 5** — LSTM seeds lifted CV by only +0.0003
  while increasing single-core inference cost; capped at 5 for budget headroom.
- The earlier repo's "CMDMamba ensemble" was aspirational (no runnable
  artifacts) and based on an unverifiable reference; it was discarded in favor
  of a model justified by the measured data structure.

## 8. Reproduction

```bash
# environment (CPU submission-faithful)
py -3.11 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
# optional GPU training env
py -3.11 -m venv .venv-gpu && .venv-gpu/Scripts/python -m pip install -r requirements-train-gpu.txt

python scripts/analyze_data.py                      # data diagnostics
python scripts/create_folds.py --output config/folds.json
python scripts/train_sequence.py --run-name gru --dev-fold 0 --device cuda   # one fold
python scripts/cv_summary.py --runs <run...>        # ensemble CV mean
python scripts/evaluate_ensemble.py --split holdout --gru <ckpt...>          # holdout
python scripts/create_submission.py --output submissions/submission.zip
python -m pytest tests -q
```
