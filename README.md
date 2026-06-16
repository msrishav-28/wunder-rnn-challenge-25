# Wunder Fund RNN Challenge — Causal GRU Solution

Portfolio reconstruction of the (finished) Wunder Fund market-state forecasting
competition: predict the next 32-feature market state from a sequence of prior
states, scored online by mean per-feature R², under a strict CPU-only inference
budget (1 core, 16 GB RAM, 60 minutes, deterministic).

🤗 **Model + weights on the Hugging Face Hub:** [msrishav/wunder-rnn-gru-ensemble](https://huggingface.co/msrishav/wunder-rnn-gru-ensemble)
— the 5 trained ensemble members are also included here under `models/submission/gru/`.

## Result

| Model | Dev CV mean R² (folds 0–3) | Notes |
| --- | ---: | --- |
| Persistence (predict current) | −0.37 | sanity baseline |
| EWMA α=0.10 | 0.217 | best stateful linear baseline |
| Ridge causal (640 feats) | 0.327 | at the linear ceiling |
| Single causal GRU (d256) | 0.389 | per-fold 0.391 / 0.394 / 0.407 / 0.363 |
| **GRU ensemble (3×d256 + 2×d384)** | **0.392** | packaged submission |

Reference: the competition's **public leaderboard #1 was 0.3920**; the finals
(private) winner scored **0.3964**. The ensemble's cross-validated 0.392 matches
the public top; it is reported as the leaderboard-comparable headline because a
single 103-sequence fold is high-variance (this model's per-fold range is
0.363–0.411), whereas the hidden test (~517 sequences) averages over regimes.

Single-core inference (measured): the 5-member ensemble runs at **4.17
ms/prediction** → ≈ **32 minutes** for the full ~517-sequence test on one CPU
core, well inside the 60-minute limit. Predictions are deterministic across runs
(verified, max abs diff 0). Untouched fold-4 holdout of the ensemble: 0.3498
(the hardest single fold; see the report for why the CV mean is the headline).

## Approach (why a GRU)

Measured data analysis (`scripts/analyze_data.py`) drove every choice:
- Features are **pre-whitened** (≈N(0,1), no heavy tails, clipped ±5.199) → MSE
  is a clean objective; predict the **level** (differencing/persistence give
  negative R²; the series is mean-reverting).
- **Linear ceiling ≈ 0.32** (VAR(p) Ridge saturates). The ~0.07 gap to the top
  is nonlinear temporal structure, so the model is a **causal unidirectional
  GRU** that emits a next-state prediction at every step and carries its hidden
  state across the sequence (O(1) per online prediction).
- Single models of every flavor (GRU d256/d384/3-layer, LSTM) cap at ≈0.389;
  the ensemble gain comes from **multi-seed + width diversity**.

See `reports/phase2_technical_report.md` for the full write-up (architecture,
training, validation, key boosters, what didn't work).

## Layout

```
solution.py                      # competition entry point (GRU-ensemble backend)
src/models/sequence_models.py    # CausalGRUForecaster (GRU/LSTM)
src/models/sequence_inference.py # O(1)/step stateful inference + checkpoint loader
src/models/ensemble_predictor.py # uniform multi-member ensemble
src/training/sequence_trainer.py # full-sequence BPTT trainer (masked loss)
src/evaluation/stepwise.py       # leak-free official-style scorer
scripts/                         # analyze_data, train_sequence, cv_summary,
                                 # ensemble_oof, stack_cv, evaluate_ensemble,
                                 # create_submission, create_folds
config/folds.json                # locked seq-grouped 5-fold split (seed 42)
models/submission/gru/*.pt       # packaged ensemble members
```

## Environments

Two pinned environments (the submission scorer is CPU-only; GPU is for fast
local training):

```bash
py -3.11 -m venv .venv     && .venv/Scripts/python     -m pip install -r requirements.txt
py -3.11 -m venv .venv-gpu && .venv-gpu/Scripts/python -m pip install -r requirements-train-gpu.txt
```

A trained model is portable: train on GPU, load on CPU with
`map_location='cpu'` — no model change, only the inference device differs.

## Reproduce

```bash
# 1. data diagnostics + folds
.venv/Scripts/python scripts/analyze_data.py
.venv/Scripts/python scripts/create_folds.py --output config/folds.json

# 2. cross-validated ensemble members (GPU; --device cpu also works)
for f in 0 1 2 3; do
  .venv-gpu/Scripts/python scripts/train_sequence.py --run-name gru_d256 --dev-fold $f \
    --device cuda --d-model 256 --epochs 70 --seed 42
done
# (repeat for seeds 123/7 and d-model 384)

# 3. cross-validated ensemble score
.venv/Scripts/python scripts/cv_summary.py --runs gru_d256_s42 gru_d256_s123 gru_d256_s7 \
    gru_d384_s42 gru_d384_s123

# 4. final submission models on all data, then package
.venv-gpu/Scripts/python scripts/train_sequence.py --run-name final_d256_s42 --train-all \
    --device cuda --d-model 256 --seed 42      # (+ other members)
.venv/Scripts/python scripts/create_submission.py --output submissions/submission.zip

# 5. tests
.venv/Scripts/python -m pytest tests -q
```

## Hardware

Trained on an NVIDIA RTX 3050 Laptop (4 GB) at ~1.8 s/epoch (~2 min/run); a
CPU-only path (AMD Ryzen 7 5800H) reproduces the same scores within ±0.0003 at
~24 s/epoch. Submission inference is single-core CPU by design.
