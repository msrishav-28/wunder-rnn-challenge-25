```
# ROLE & OBJECTIVE
You are a world-class Quantitative Researcher specializing in market microstructure and high-frequency trading, competing for first place ($13,600 prize) in the Wunder Fund RNN Challenge. Your goal is to build a production-grade, CPU-optimized market state prediction system that maximizes R² coefficient across 32 anonymized market features.

# COMPETITION CONSTRAINTS
- **Hardware**: CPU-only (no GPU), 16GB RAM, 1 core
- **Time limit**: 60 minutes inference on ~500 test sequences (7.2ms per prediction budget)
- **Data**: 517 training sequences, each with 1,000 timesteps, 32 features per state
- **Prohibited**: External data, pre-trained models, internet access
- **Metric**: Average R² across all 32 features (higher is better)
- **Warm-up**: First 100 steps per sequence for regime inference (not scored)
- **Predictions**: Steps 101-1000 require predictions

# WINNING STRATEGY FRAMEWORK

## Phase 1: Feature Engineering (This Wins Competitions)
Create ~320 features from 32 base features using market microstructure principles:

**Temporal Features**:
- Lag features: t-1, t-2, t-5, t-10, t-20, t-50, t-100
- Rolling statistics: mean/std/min/max for windows [5, 10, 20, 50, 100]
- Exponential moving averages: EMA(5), EMA(20), EMA(50)
- Rate of change: (current - lag_n) / lag_n for n=[1,5,10,20]
- Acceleration: first and second derivatives
- Autocorrelation at lags [1, 5, 10, 20]

**Regime Detection Features** (from warm-up period 0-99):
- Volatility regime: std(last_100) / global_std
- Trend strength: linear_regression_slope(last_100)
- Mean reversion score: distance_from_MA(50) / std(50)
- Market state clustering: K-means cluster assignment (fit on all train sequences)

**Cross-Feature Interactions**:
- Feature crosses for top-10 most correlated feature pairs
- Ratio features: feat_i / feat_j for important pairs
- Z-score normalization per sequence

**Statistical Transforms**:
- Quantile transforms (handle non-Gaussian distributions)
- Box-Cox transforms for variance stabilization
- Fourier features to capture periodicity

## Phase 2: Model Ensemble (5 Diverse Architectures)

**Model 1: Lightweight GRU** (Primary model - fastest inference)
```
import intel_extension_for_pytorch as ipex
model = nn.GRU(input_size=320, hidden_size=64, num_layers=2, batch_first=True)
model = ipex.optimize(model, dtype=torch.bfloat16)
# Target: 2ms inference per sequence
```

**Model 2: LightGBM Gradient Boosting** (Tree ensemble - CPU optimized)
- Train 32 separate models (one per target feature)
- Params: num_leaves=31, max_depth=8, n_estimators=500, learning_rate=0.05
- Target: 0.5ms inference per sequence

**Model 3: Ridge Regression with Polynomial Features**
- Degree-2 polynomials on top-50 most important features (from LightGBM)
- Handles multicollinearity, extremely fast on CPU
- Target: 0.1ms inference per sequence

**Model 4: Temporal Convolutional Network**
- 1D convolutions with dilations [1,2,4,8,16]
- 64 filters per layer, kernel_size=3
- Parallelizes well on CPU, captures multi-scale patterns
- Target: 3ms inference per sequence

**Model 5: Bidirectional GRU + Attention**
- Same as Model 1 but bidirectional with scaled dot-product attention
- Use for high-uncertainty sequences only
- Target: 4ms inference per sequence

## Phase 3: Training Strategy (CPU-Optimized)

**Intel CPU Acceleration**:
```
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,compact,1,0
pip install intel-extension-for-pytorch onednn
```

**5-Fold Cross-Validation by seq_ix**:
- Split sequences (not timesteps) into 5 folds
- Generate out-of-fold predictions for all 517 sequences
- Use OOF predictions to optimize ensemble weights

**Per-Feature Model Selection**:
- Compute R² for each of 32 features on validation set
- Select best-performing model per feature
- Create heterogeneous ensemble (different models for different features)

**Training Schedule** (CPU timings):
1. LightGBM: 2-3 hours (all 32 models)
2. Ridge: 30 minutes
3. GRU: 15-20 hours (with Intel extensions + bfloat16)
4. TCN: 12-15 hours
5. BiGRU: 18-22 hours

Total: ~48-60 hours on modern multi-core CPU

## Phase 4: Ensemble Optimization

**Weighted Ensemble with Per-Feature Weights**:
```
# For each of 32 features, optimize weights across 5 models
# Use validation R² as objective
# Constraints: weights >= 0, sum(weights) = 1 per feature

def ensemble_predict(models, X, weights):
    predictions = np.zeros((len(X), 32))
    for i in range(32):
        feature_preds = [model.predict(X)[:, i] for model in models]
        predictions[:, i] = np.average(feature_preds, weights=weights[i], axis=0)
    return predictions
```

**Meta-Learning (Stacking)**:
- Train Ridge regression on OOF predictions as meta-learner
- Input: [Model1_pred, Model2_pred, ..., Model5_pred, engineered_features]
- Output: Final 32-feature prediction

**Negative R² Prevention**:
```
# If model has R² < 0.1 on validation, blend with feature mean
if val_r2[feature_idx] < 0.1:
    pred = 0.7 * model_pred + 0.3 * train_mean[feature_idx]
```

## Phase 5: CPU Inference Optimization (Critical for 60min limit)

**ONNX Runtime Conversion**:
```
import onnxruntime as ort
torch.onnx.export(model, dummy_input, "model.onnx")
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
# Expected: 2-3x speedup over PyTorch on CPU
```

**INT8 Quantization**:
```
from onnxruntime.quantization import quantize_dynamic
quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)
# Expected: 2-3x additional speedup, 4x size reduction
```

**Batch Processing**:
- Process multiple sequences in parallel using NumPy vectorization
- Leverage CPU multi-threading via BLAS libraries

**Feature Engineering Pipeline Optimization**:
- Pre-compute rolling statistics using efficient online algorithms
- Cache global statistics (means, stds) computed on training set
- Use NumPy strides for lag feature extraction (zero-copy)

## Phase 6: Solution Structure

**solution.py Template**:
```
import numpy as np
import onnxruntime as ort
import lightgbm as lgb
import joblib
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Load optimized models
        self.gru_session = ort.InferenceSession("gru_int8.onnx")
        self.lgb_models = [lgb.Booster(model_file=f"lgb_{i}.txt") for i in range(32)]
        self.ridge = joblib.load("ridge.pkl")
        self.tcn_session = ort.InferenceSession("tcn_int8.onnx")
        
        # Load preprocessing artifacts
        self.scaler = joblib.load("scaler.pkl")
        self.feature_means = np.load("train_means.npy")
        self.feature_stds = np.load("train_stds.npy")
        self.ensemble_weights = np.load("ensemble_weights.npy")  # Shape: (32, 5)
        
        # State tracking
        self.sequence_buffer = []
        self.current_seq = -1
        self.hidden_state = None
        self.regime_features = None
    
    def _engineer_features(self, data_point):
        # Implement 320-feature engineering pipeline
        # Return: (320,) array
        pass
    
    def _detect_regime(self, warmup_states):
        # Compute regime features from first 100 states
        # Return: volatility_regime, trend_strength, mean_reversion_score
        pass
    
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Handle sequence reset
        if data_point.seq_ix != self.current_seq:
            self.sequence_buffer = []
            self.hidden_state = None
            self.current_seq = data_point.seq_ix
            self.regime_features = None
        
        # Buffer states
        self.sequence_buffer.append(data_point.state)
        
        # Compute regime features after warm-up
        if len(self.sequence_buffer) == 100:
            self.regime_features = self._detect_regime(np.array(self.sequence_buffer))
        
        if not data_point.need_prediction:
            return None
        
        # Feature engineering
        features = self._engineer_features(data_point)
        
        # Ensemble prediction
        predictions = np.zeros((5, 32))
        predictions = self._gru_predict(features)
        predictions = self._lgb_predict(features)[1]
        predictions = self._ridge_predict(features)[2]
        predictions = self._tcn_predict(features)[3]
        predictions = self._bigru_predict(features)[4]
        
        # Weighted ensemble per feature
        final_pred = np.sum(predictions.T * self.ensemble_weights, axis=1)
        
        return final_pred
```

# IMPLEMENTATION REQUIREMENTS

1. **Deterministic Execution**: Set all random seeds (np.random.seed, torch.manual_seed, random.seed)
2. **Memory Efficiency**: Keep sequence buffer size bounded, clear after each sequence
3. **Error Handling**: Catch NaN/Inf in predictions, replace with feature means
4. **Logging**: Track inference time per model for optimization
5. **Validation**: Achieve local R² > 0.85 before submission

# DELIVERABLES

Generate complete, production-ready code for:
1. `feature_engineering.py` - Feature extraction pipeline (320 features)
2. `train_models.py` - Train 5 models with 5-fold CV, generate OOF predictions
3. `optimize_ensemble.py` - Optimize per-feature weights, train meta-learner
4. `export_models.py` - Convert to ONNX, quantize, optimize for CPU
5. `solution.py` - Final submission file with PredictionModel class
6. `config.yaml` - Hyperparameters, paths, random seeds
7. `requirements.txt` - Exact dependencies with versions
8. `README.md` - Setup instructions, training commands, validation results

# SUCCESS CRITERIA
- Local validation R² > 0.85 (target: 0.90+)
- Inference time < 45 minutes (25% safety margin below 60min limit)
- Code runs deterministically (same results on repeated runs)
- Memory usage < 12GB (leaves 4GB buffer)
- All models export to ONNX successfully
- Per-feature R² all positive (no negative R² on any feature)

Build this winning solution with extreme attention to CPU optimization, feature quality, and ensemble diversity. Focus on practical market microstructure insights over academic complexity.
```
