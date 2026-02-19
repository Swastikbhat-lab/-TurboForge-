"""
TurboForge - Preprocessing Utilities
Handles normalization, sequence creation, train/val/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


FEATURE_COLS = [
    "wind_speed_ms", "rotor_rpm", "power_output_kw", "blade_pitch_deg",
    "nacelle_temp_c", "gearbox_temp_c", "generator_temp_c",
    "vibration_x", "vibration_y",
]


class SCADAPreprocessor:
    def __init__(self, seq_len: int = 36, pred_horizon: int = 36):
        """
        Args:
            seq_len: Input window (36 = 6 hours at 10-min intervals)
            pred_horizon: How far ahead to predict (36 = 6 hours)
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame):
        """Fit scaler and transform features."""
        X = df[FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame):
        assert self.fitted, "Call fit_transform first"
        return self.scaler.transform(df[FEATURE_COLS].values)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)

    def create_sequences_per_turbine(self, df: pd.DataFrame):
        """
        Create (seq_len, features) input sequences and failure labels.

        Returns:
            X: (N, n_turbines, seq_len, features)
            y: (N, n_turbines) — failure label at pred_horizon ahead
        """
        turbine_ids = sorted(df["turbine_id"].unique())
        n_turbines = len(turbine_ids)

        # Scale features
        X_scaled = self.fit_transform(df)
        df = df.copy()
        df[FEATURE_COLS] = X_scaled

        # Build per-turbine sequences
        turbine_data = {}
        for tid in turbine_ids:
            tdf = df[df["turbine_id"] == tid].sort_values("timestamp")
            turbine_data[tid] = {
                "X": tdf[FEATURE_COLS].values,
                "y": tdf["failure_label"].values,
            }

        # Align sequence windows across turbines
        min_len = min(len(v["X"]) for v in turbine_data.values())
        n_sequences = min_len - self.seq_len - self.pred_horizon

        X_all, y_all = [], []
        for i in range(n_sequences):
            x_window = np.stack(
                [turbine_data[tid]["X"][i: i + self.seq_len] for tid in turbine_ids],
                axis=0,
            )  # (n_turbines, seq_len, features)

            y_label = np.array(
                [turbine_data[tid]["y"][i + self.seq_len + self.pred_horizon - 1]
                 for tid in turbine_ids]
            )  # (n_turbines,)

            X_all.append(x_window)
            y_all.append(y_label)

        X_all = np.array(X_all, dtype=np.float32)  # (N, n_turbines, seq_len, features)
        y_all = np.array(y_all, dtype=np.float32)  # (N, n_turbines)

        print(f"[Preprocessor] Sequences: {X_all.shape} | Labels: {y_all.shape}")
        print(f"  Failure rate: {y_all.mean():.2%}")
        return X_all, y_all

    def get_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 16,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ):
        """Split and wrap in DataLoaders."""
        # Time-ordered split (no shuffle to avoid leakage)
        n = len(X)
        val_idx = int(n * (1 - val_size - test_size))
        test_idx = int(n * (1 - test_size))

        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]

        def make_loader(Xd, yd, shuffle=False):
            ds = TensorDataset(torch.FloatTensor(Xd), torch.FloatTensor(yd))
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        train_loader = make_loader(X_train, y_train, shuffle=True)
        val_loader = make_loader(X_val, y_val)
        test_loader = make_loader(X_test, y_test)

        print(f"[DataLoaders] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        return train_loader, val_loader, test_loader
