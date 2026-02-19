"""
TurboForge - Cross-Turbine Transformer
Models spatial dependencies across 50 turbines for failure prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TurbineEncoder(nn.Module):
    def __init__(self, feature_dim, d_model, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.transformer(x)
        return self.pool(x.transpose(1, 2)).squeeze(-1)


class CrossTurbineAttention(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class TurboForgeTransformer(nn.Module):
    def __init__(self, n_turbines=50, feature_dim=9, d_model=128, temporal_heads=4, spatial_heads=8, temporal_layers=2, spatial_layers=2):
        super().__init__()
        self.n_turbines = n_turbines
        self.turbine_encoder = TurbineEncoder(feature_dim, d_model, temporal_heads, temporal_layers)
        self.cross_turbine = CrossTurbineAttention(d_model, spatial_heads, spatial_layers)
        self.failure_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d_model//2, 1))
        self.coordination_head = nn.Sequential(nn.Linear(d_model*n_turbines, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1), nn.Sigmoid())
        print(f"[TurboForgeTransformer] Initialized")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        batch_size = x.size(0)
        turbine_embs = torch.stack([self.turbine_encoder(x[:, t, :, :]) for t in range(self.n_turbines)], dim=1)
        attended = self.cross_turbine(turbine_embs)
        failure_probs = torch.sigmoid(self.failure_head(attended).squeeze(-1))
        coordination_score = self.coordination_head(attended.reshape(batch_size, -1))
        return failure_probs, coordination_score


class TurboForgeTrainer:
    def __init__(self, model, lr=1e-4, device="auto", pos_weight=10.0):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.pos_weight = torch.tensor([pos_weight]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def _focal_loss(self, pred, target, gamma=2.0):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        return ((1 - pt) ** gamma * bce).mean()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            failure_probs, _ = self.model(X_batch)
            loss = self.criterion(failure_probs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            failure_probs, _ = self.model(X_batch)
            preds = (failure_probs > 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
        try:
            auc = roc_auc_score(all_labels.flatten(), all_preds.flatten())
        except ValueError:
            auc = 0.0
        return {"accuracy": acc, "roc_auc": auc}

    def fit(self, train_loader, val_loader, epochs=50):
        best_acc = 0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save(self.model.state_dict(), "best_turboforge.pt")
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch [{epoch:03d}/{epochs}] | Loss: {train_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | AUC: {metrics['roc_auc']:.4f}")
        print(f"\n[Training Complete] Best Val Accuracy: {best_acc:.4f}")
        return best_acc
