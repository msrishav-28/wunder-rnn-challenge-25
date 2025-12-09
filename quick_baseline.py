"""
Quick Baseline Solution - Fast R² achiever
Focuses on LightGBM + feature engineering for quick validation
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'competition_package'))
from utils import DataPoint, ScorerStepByStep

from feature_engineering import FeatureEngineer


class QuickBaselineModel:
    def __init__(self, use_pretrained=False):
        self.current_seq_ix = None
        self.sequence_buffer = []
        self.feature_engineer = FeatureEngineer(n_features=32, n_engineered=320)
        self.lgb_models = None  # Will store 32 LightGBM models
        self.scaler = StandardScaler()
        self.feature_means = None
        self.use_pretrained = use_pretrained
        
        if use_pretrained and os.path.exists('models/feature_engineer.pkl'):
            self.feature_engineer = joblib.load('models/feature_engineer.pkl')
            self.lgb_models = joblib.load('models/lgb_ensemble.pkl')
    
    def train_on_data(self, data_path):
        """Train LightGBM models on full training data"""
        print("Loading training data...")
        df = pd.read_parquet(data_path)
        
        # Group by sequence and extract states
        sequences = []
        for seq_ix in sorted(df['seq_ix'].unique()):
            seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            states = seq_data.iloc[:, 3:].values  # Get feature columns (32 features)
            sequences.append(states)
        
        print(f"Loaded {len(sequences)} sequences")
        
        # Fit feature engineer
        print("Fitting feature engineer...")
        self.feature_engineer.fit(sequences)
        
        # Generate engineered features for all sequences
        print("Generating engineered features...")
        X_list = []
        y_list = []
        
        for seq_idx, seq in enumerate(sequences):
            if seq_idx % 100 == 0:
                print(f"  Processing sequence {seq_idx}/{len(sequences)}")
            
            # Process each state and build X, y pairs
            buffer = []
            for t in range(len(seq)):
                buffer.append(seq[t])
                
                # We need (current_features -> next_state)
                if t > 0:  # Skip first timestep
                    features = self.feature_engineer.engineer_features(buffer[:-1], t-1)
                    X_list.append(features)
                    y_list.append(seq[t])  # Next state is the target
        
        X_array = np.array(X_list, dtype=np.float32)
        y_array = np.array(y_list, dtype=np.float32)
        
        print(f"Training data shape: {X_array.shape}")
        print(f"Target data shape: {y_array.shape}")
        
        # Train LightGBM models (one per feature)
        self.lgb_models = []
        print("Training LightGBM models...")
        
        for feature_idx in range(32):
            print(f"  Feature {feature_idx}/31...", end=" ", flush=True)
            
            y_feature = y_array[:, feature_idx]
            
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X_array, y_feature)
            self.lgb_models.append(model)
            print("✓")
        
        # Store feature means for fallback
        self.feature_means = np.mean(y_array, axis=0)
        
        # Save models
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.feature_engineer, 'models/feature_engineer.pkl')
        joblib.dump(self.lgb_models, 'models/lgb_ensemble.pkl')
        joblib.dump(self.feature_means, 'models/feature_means.pkl')
        print("Models saved!")
    
    def predict(self, data_point: DataPoint) -> np.ndarray:
        """Predict next state given current data point"""
        
        # Reset sequence buffer when sequence changes
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_buffer = []
        
        # Add current state to buffer
        self.sequence_buffer.append(data_point.state.copy())
        
        # Return None if no prediction is needed
        if not data_point.need_prediction:
            return None
        
        # Engineer features from sequence buffer
        features = self.feature_engineer.engineer_features(self.sequence_buffer, 
                                                           len(self.sequence_buffer) - 1)
        
        # Make predictions using LightGBM ensemble
        if self.lgb_models is None:
            return self.feature_means.copy()
        
        predictions = np.zeros(32, dtype=np.float32)
        for feat_idx, model in enumerate(self.lgb_models):
            try:
                pred = model.predict(features.reshape(1, -1))[0]
                predictions[feat_idx] = pred
            except Exception as e:
                predictions[feat_idx] = self.feature_means[feat_idx]
        
        return predictions


if __name__ == "__main__":
    # Train or load model
    model = QuickBaselineModel(use_pretrained=False)
    
    # Train on full dataset
    model.train_on_data('competition_package/datasets/train.parquet')
    
    # Evaluate on test set
    scorer = ScorerStepByStep('competition_package/datasets/train.parquet')
    
    print("\nEvaluating on full training dataset...")
    scores = scorer.score(model)
    
    print("\n=== RESULTS ===")
    for feature, r2 in sorted(scores.items()):
        if feature != 'mean_r2':
            print(f"Feature {feature}: R² = {r2:.4f}")
    print(f"\nMean R² = {scores['mean_r2']:.4f}")
