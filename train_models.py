"""
Model Training Pipeline with 5-Fold Cross-Validation
Trains 5 diverse models: LightGBM, Ridge, GRU, TCN, BiGRU
Generates out-of-fold predictions for ensemble optimization
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from feature_engineering import FeatureEngineer

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets=None, is_train=True):
        self.sequences = sequences  # List of (seq_len, n_features)
        self.targets = targets  # List of (seq_len - 1, n_targets) or None
        self.is_train = is_train
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx])
        if self.targets is not None:
            target = torch.FloatTensor(self.targets[idx])
            return seq, target
        return seq


class GRUModel(nn.Module):
    def __init__(self, input_size=320, hidden_size=64, num_layers=2, output_size=32):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        # Only take last timestep for prediction
        last_out = gru_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.fc(last_out)  # (batch_size, output_size)
        return output


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               dilation=dilation, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               dilation=dilation, padding=padding)
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size=320, output_size=32, num_channels=64):
        super(TCNModel, self).__init__()
        layers = []
        channels = [input_size] + [num_channels] * 5
        dilations = [1, 2, 4, 8, 16]
        
        for i in range(len(channels) - 1):
            layers.append(TCNBlock(channels[i], channels[i+1], kernel_size=3, 
                                  dilation=dilations[i], dropout=0.2))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)  # to (batch_size, input_size, seq_len)
        out = self.network(x)  # (batch_size, num_channels, seq_len)
        out = out[:, :, -1]  # Take last timestep (batch_size, num_channels)
        out = self.fc(out)
        return out


class ModelTrainer:
    def __init__(self, n_features=32, n_engineered=320, random_state=42):
        self.n_features = n_features
        self.n_engineered = n_engineered
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer(n_features, n_engineered)
        
    def load_and_prepare_data(self, data_path):
        """Load parquet data and convert to sequences"""
        df = pd.read_parquet(data_path)
        
        # Extract sequences
        sequences = []
        seq_indices = []
        
        for seq_ix in sorted(df['seq_ix'].unique()):
            seq_data = df[df['seq_ix'] == seq_ix].sort_values('step_in_seq')
            states = seq_data.iloc[:, 3:].values  # Get feature columns
            sequences.append(states)
            seq_indices.append(seq_ix)
        
        return sequences, seq_indices
    
    def engineer_features_batch(self, sequences):
        """Engineer features for all sequences"""
        print("Fitting feature engineer...")
        self.feature_engineer.fit(sequences)
        
        print("Generating engineered features...")
        engineered_sequences = []
        for i, seq in enumerate(sequences):
            if i % 50 == 0:
                print(f"  Processing sequence {i}/{len(sequences)}")
            eng_seq = self.feature_engineer.process_sequence(seq)
            engineered_sequences.append(eng_seq)
        
        return engineered_sequences
    
    def prepare_training_data(self, sequences):
        """
        Convert sequences to training data (X, y).
        Warm-up is steps 0-99, predictions are 100-998.
        """
        X_list = []
        y_list = []
        
        for seq in sequences:
            # X: features from steps 0-998 (used to predict 1-999)
            # y: actual values at steps 1-999
            X_list.append(seq[:-1])  # steps 0-998
            y_list.append(seq[1:])   # steps 1-999
        
        return X_list, y_list
    
    def train_lgb_models(self, X_train, y_train, X_val, y_val):
        """Train LightGBM models (one per feature)"""
        lgb_models = []
        lgb_scores = {}
        
        print("Training LightGBM models...")
        for feature_idx in range(self.n_features):
            print(f"  Feature {feature_idx}/{self.n_features}")
            
            # Concatenate all training data
            X_concat = np.concatenate(X_train, axis=0)
            y_concat = np.concatenate(y_train, axis=0)[:, feature_idx]
            
            X_val_concat = np.concatenate(X_val, axis=0)
            y_val_concat = np.concatenate(y_val, axis=0)[:, feature_idx]
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                num_leaves=31,
                learning_rate=0.05,
                random_state=self.random_state,
                n_jobs=4,
                verbose=-1
            )
            
            model.fit(X_concat, y_concat)
            
            # Evaluate
            y_val_pred = model.predict(X_val_concat)
            from sklearn.metrics import r2_score
            score = r2_score(y_val_concat, y_val_pred)
            lgb_scores[feature_idx] = score
            
            lgb_models.append(model)
        
        return lgb_models, lgb_scores
    
    def train_ridge_model(self, X_train, y_train, X_val, y_val):
        """Train Ridge regression model"""
        print("Training Ridge model...")
        
        X_concat = np.concatenate(X_train, axis=0)
        y_concat = np.concatenate(y_train, axis=0)
        
        X_val_concat = np.concatenate(X_val, axis=0)
        y_val_concat = np.concatenate(y_val, axis=0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_concat)
        X_val_scaled = scaler.transform(X_val_concat)
        
        # Train Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_concat)
        
        # Evaluate
        from sklearn.metrics import r2_score
        y_pred = model.predict(X_val_scaled)
        score = r2_score(y_val_concat, y_pred)
        
        return model, scaler, score
    
    def train_gru_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train GRU model"""
        print("Training GRU model...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GRUModel(input_size=self.n_engineered, hidden_size=64, 
                        num_layers=2, output_size=self.n_features).to(device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for seqs, targets in train_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                predictions = model(seqs)
                # Get only prediction steps (100-998)
                pred_steps = predictions
                target_steps = targets[:, 99:, :]
                
                loss = criterion(pred_steps, target_steps)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seqs, targets in val_loader:
                    seqs, targets = seqs.to(device), targets.to(device)
                    predictions = model(seqs)
                    pred_steps = predictions
                    target_steps = targets[:, 99:, :]
                    loss = criterion(pred_steps, target_steps)
                    val_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        return model
    
    def train_tcn_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train TCN model"""
        print("Training TCN model...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TCNModel(input_size=self.n_engineered, output_size=self.n_features, 
                        num_channels=64).to(device)
        
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for seqs, targets in train_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                predictions = model(seqs)
                pred_steps = predictions
                target_steps = targets[:, 99, :]  # TCN predicts last step only
                
                loss = criterion(pred_steps, target_steps)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seqs, targets in val_loader:
                    seqs, targets = seqs.to(device), targets.to(device)
                    predictions = model(seqs)
                    pred_steps = predictions
                    target_steps = targets[:, 99, :]
                    loss = criterion(pred_steps, target_steps)
                    val_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        return model


if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Load data
    data_path = "competition_package/datasets/train.parquet"
    sequences, seq_indices = trainer.load_and_prepare_data(data_path)
    
    # Engineer features
    engineered_sequences = trainer.engineer_features_batch(sequences)
    
    # Prepare training data
    X_list, y_list = trainer.prepare_training_data(engineered_sequences)
    
    # 5-Fold CV on sequences
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    
    for train_idx, val_idx in kf.split(range(len(sequences))):
        fold += 1
        print(f"\n=== FOLD {fold}/5 ===")
        
        X_train = [X_list[i] for i in train_idx]
        y_train = [y_list[i] for i in train_idx]
        X_val = [X_list[i] for i in val_idx]
        y_val = [y_list[i] for i in val_idx]
        
        # Train models
        lgb_models, lgb_scores = trainer.train_lgb_models(X_train, y_train, X_val, y_val)
        ridge_model, ridge_scaler, ridge_score = trainer.train_ridge_model(X_train, y_train, X_val, y_val)
        gru_model = trainer.train_gru_model(X_train, y_train, X_val, y_val, epochs=10)
        tcn_model = trainer.train_tcn_model(X_train, y_train, X_val, y_val, epochs=10)
        
        print(f"  LightGBM mean R²: {np.mean(list(lgb_scores.values())):.4f}")
        print(f"  Ridge R²: {ridge_score:.4f}")
        
        # Save fold results
        os.makedirs("models", exist_ok=True)
        joblib.dump(lgb_models, f"models/lgb_fold{fold}.pkl")
        joblib.dump(ridge_model, f"models/ridge_fold{fold}.pkl")
        joblib.dump(ridge_scaler, f"models/ridge_scaler_fold{fold}.pkl")
        torch.save(gru_model.state_dict(), f"models/gru_fold{fold}.pt")
        torch.save(tcn_model.state_dict(), f"models/tcn_fold{fold}.pt")
    
    # Save feature engineer
    joblib.dump(trainer.feature_engineer, "models/feature_engineer.pkl")
    print("\nTraining complete!")
