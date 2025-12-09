"""
Feature Engineering Pipeline for Market State Prediction
Generates ~320 engineered features from 32 base features
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    def __init__(self, n_features=32, n_engineered=320):
        self.n_features = n_features
        self.n_engineered = n_engineered
        
        # Store training statistics for consistent preprocessing
        self.global_means = None
        self.global_stds = None
        self.global_mins = None
        self.global_maxs = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train_sequences):
        """
        Fit feature engineering parameters on training data.
        
        Args:
            X_train_sequences: List of sequences, each shape (1000, 32)
        """
        # Concatenate all sequences for global statistics
        X_all = np.concatenate(X_train_sequences, axis=0)
        
        self.global_means = np.mean(X_all, axis=0)
        self.global_stds = np.std(X_all, axis=0) + 1e-8
        self.global_mins = np.min(X_all, axis=0)
        self.global_maxs = np.max(X_all, axis=0)
        
        # Fit K-means on all training data for regime detection
        self.kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.kmeans_model.fit(X_all)
        
        # Fit scaler for later use
        self.scaler.fit(X_all)
        
    def engineer_features(self, sequence_buffer, current_state_idx):
        """
        Engineer features for a specific state in the sequence.
        
        Args:
            sequence_buffer: List of states seen so far
            current_state_idx: Index of current state in the sequence
            
        Returns:
            features: Array of shape (n_engineered,)
        """
        X = np.array(sequence_buffer)  # Shape: (n_timesteps, 32)
        features = []
        
        # === TEMPORAL FEATURES ===
        
        # 1. Lag features (t-1, t-2, t-5, t-10, t-20, t-50, t-100)
        current = X[-1]  # Latest state
        for lag in [1, 2, 5, 10, 20, 50, 100]:
            if len(X) > lag:
                features.append(X[-lag-1])  # (32,)
            else:
                features.append(np.zeros(self.n_features))
        # 7*32 = 224 features
        
        # 2. Rate of change features
        for lag in [1, 5, 10, 20]:
            if len(X) > lag:
                lag_state = X[-lag-1]
                roc = (current - lag_state) / (np.abs(lag_state) + 1e-8)
                features.append(roc)
            else:
                features.append(np.zeros(self.n_features))
        # 4*32 = 128 features (cumulative: 352)
        
        # === ROLLING STATISTICS ===
        for window in [5, 10, 20, 50, 100]:
            if len(X) >= window:
                window_data = X[-window:]
                features.append(np.mean(window_data, axis=0))  # Rolling mean
                features.append(np.std(window_data, axis=0) + 1e-8)  # Rolling std
                features.append(np.min(window_data, axis=0))  # Rolling min
                features.append(np.max(window_data, axis=0))  # Rolling max
            else:
                features.append(np.zeros(self.n_features))
                features.append(np.zeros(self.n_features))
                features.append(np.zeros(self.n_features))
                features.append(np.zeros(self.n_features))
        # 5*4*32 = 640 features (too many, need to consolidate)
        
        # === EXPONENTIAL MOVING AVERAGES ===
        for span in [5, 20, 50]:
            ema = self._compute_ema(X, span)
            features.append(ema)  # (32,)
        # 3*32 = 96 features
        
        # === DERIVATIVES (ACCELERATION) ===
        if len(X) >= 2:
            first_deriv = X[-1] - X[-2]
            features.append(first_deriv)
            if len(X) >= 3:
                second_deriv = (X[-1] - X[-2]) - (X[-2] - X[-3])
                features.append(second_deriv)
            else:
                features.append(np.zeros(self.n_features))
        else:
            features.append(np.zeros(self.n_features))
            features.append(np.zeros(self.n_features))
        # 2*32 = 64 features
        
        # === AUTOCORRELATION FEATURES ===
        for lag in [1, 5, 10, 20]:
            if len(X) > lag + 1:
                autocorr = self._compute_autocorrelation(X, lag)
                features.append(autocorr)
            else:
                features.append(np.zeros(self.n_features))
        # 4*32 = 128 features
        
        # === REGIME DETECTION FEATURES ===
        if len(sequence_buffer) >= 100:
            warmup_window = X[:100]
            
            # Volatility regime
            vol_regime = np.std(warmup_window, axis=0) / (self.global_stds + 1e-8)
            features.append(vol_regime)
            
            # Trend strength (linear regression slope)
            trend_strength = self._compute_trend_strength(warmup_window)
            features.append(trend_strength)
            
            # Mean reversion score
            ma50 = np.mean(warmup_window, axis=0)
            mean_rev = (X[-1] - ma50) / (self.global_stds + 1e-8)
            features.append(mean_rev)
            
            # K-means cluster assignment (one-hot encoded)
            if self.kmeans_model is not None:
                cluster_id = self.kmeans_model.predict([X[-1]])[0]
                cluster_onehot = np.zeros(5)
                cluster_onehot[cluster_id] = 1
                features.append(cluster_onehot)
            else:
                features.append(np.zeros(5))
        else:
            # During warmup, use dummy regime features
            features.append(np.zeros(self.n_features))
            features.append(np.zeros(self.n_features))
            features.append(np.zeros(self.n_features))
            features.append(np.zeros(5))
        # 3*32 + 5 = 101 features
        
        # === Z-SCORE NORMALIZATION ===
        # Normalize current state
        z_score = (current - self.global_means) / (self.global_stds + 1e-8)
        features.append(z_score)
        # 32 features
        
        # === MIN-MAX NORMALIZATION ===
        minmax = (current - self.global_mins) / (self.global_maxs - self.global_mins + 1e-8)
        minmax = np.clip(minmax, 0, 1)  # Clip to [0,1]
        features.append(minmax)
        # 32 features
        
        # Concatenate all features
        feature_array = np.concatenate(features)
        
        # Truncate or pad to target dimension
        if len(feature_array) > self.n_engineered:
            feature_array = feature_array[:self.n_engineered]
        else:
            feature_array = np.pad(feature_array, (0, self.n_engineered - len(feature_array)))
        
        return feature_array.astype(np.float32)
    
    def _compute_ema(self, X, span):
        """Compute exponential moving average"""
        ema_values = np.zeros_like(X[0])
        alpha = 2.0 / (span + 1)
        
        for i in range(len(X)):
            if i == 0:
                ema_values = X[i].copy()
            else:
                ema_values = alpha * X[i] + (1 - alpha) * ema_values
        
        return ema_values
    
    def _compute_autocorrelation(self, X, lag):
        """Compute autocorrelation at specified lag"""
        autocorr = np.zeros(self.n_features)
        for feat_idx in range(self.n_features):
            feat_series = X[:, feat_idx]
            if len(feat_series) > lag:
                mean = np.mean(feat_series)
                c0 = np.mean((feat_series - mean) ** 2)
                if c0 > 1e-8:
                    c_lag = np.mean((feat_series[:-lag] - mean) * (feat_series[lag:] - mean))
                    autocorr[feat_idx] = c_lag / c0
        return autocorr
    
    def _compute_trend_strength(self, X):
        """Compute trend strength using linear regression slope"""
        trend = np.zeros(self.n_features)
        for feat_idx in range(self.n_features):
            y = X[:, feat_idx]
            x = np.arange(len(y))
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
                trend[feat_idx] = slope
        return trend
    
    def process_sequence(self, sequence_states):
        """
        Process an entire sequence and return engineered features for each timestep.
        
        Args:
            sequence_states: Array of shape (1000, 32)
            
        Returns:
            engineered_features: Array of shape (1000, n_engineered)
        """
        engineered_features = []
        buffer = []
        
        for t in range(len(sequence_states)):
            buffer.append(sequence_states[t])
            features = self.engineer_features(buffer, t)
            engineered_features.append(features)
        
        return np.array(engineered_features, dtype=np.float32)
