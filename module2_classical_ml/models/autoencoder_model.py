
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from module2_classical_ml.config.defaults import (
    AUTOENCODER_BATCH_SIZE,
    AUTOENCODER_EARLY_STOPPING_PATIENCE,
    AUTOENCODER_EPOCHS,
    AUTOENCODER_LEARNING_RATE,
    AUTOENCODER_RECONSTRUCTION_PERCENTILE,
)
from module2_classical_ml.features.feature_pipeline import FEATURE_DIM


class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:

    def __init__(self, input_dim: int = FEATURE_DIM) -> None:
        self._net = _AutoencoderNet(input_dim)
        self._threshold_99: float = 1.0
        self._is_trained = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net.to(self._device)
        self._scaler = StandardScaler()

    def fit(
        self,
        X_normal: np.ndarray,
        epochs: int = AUTOENCODER_EPOCHS,
        batch_size: int = AUTOENCODER_BATCH_SIZE,
        lr: float = AUTOENCODER_LEARNING_RATE,
        patience: int = AUTOENCODER_EARLY_STOPPING_PATIENCE,
    ) -> None:

        X_scaled = self._scaler.fit_transform(X_normal)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        dataset = TensorDataset(X_t)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self._net.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        self._net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self._net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break


        self._net.eval()
        with torch.no_grad():
            recon = self._net(X_t)
            errors = torch.mean((X_t - recon) ** 2, dim=1).cpu().numpy()
        self._threshold_99 = float(np.percentile(errors, AUTOENCODER_RECONSTRUCTION_PERCENTILE))
        self._is_trained = True

    def score(self, x: np.ndarray) -> float:
       
        if not self._is_trained:
            return 0.0

        self._net.eval()
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            recon = self._net(x_t)
            mse = torch.mean((x_t - recon) ** 2).item()

        return float(min(mse / max(self._threshold_99, 1e-9), 1.0))

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            return np.zeros(len(X))

        self._net.eval()
        X_scaled = self._scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            recon = self._net(X_t)
            errors = torch.mean((X_t - recon) ** 2, dim=1).cpu().numpy()
        return np.clip(errors / max(self._threshold_99, 1e-9), 0.0, 1.0)

    def save(self, path: Path) -> None:
        torch.save({
            "state_dict": self._net.state_dict(),
            "threshold_99": self._threshold_99,
            "input_dim": self._net.encoder[0].in_features,
            "scaler": self._scaler,
        }, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        input_dim = ckpt["input_dim"]
        self._net = _AutoencoderNet(input_dim).to(self._device)
        self._net.load_state_dict(ckpt["state_dict"])
        self._threshold_99 = ckpt["threshold_99"]
        self._scaler = ckpt["scaler"]
        self._is_trained = True
