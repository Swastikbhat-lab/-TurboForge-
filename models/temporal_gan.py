"""
TurboForge - Temporal GAN (TimeGAN-inspired)
Generates synthetic SCADA time series to augment training data and
enable counterfactual "what-if" scenario generation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────
# Model Components
# ─────────────────────────────────────────────

class TemporalGenerator(nn.Module):
    """GRU-based generator: noise → synthetic SCADA sequences."""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        x = self.fc_in(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.gru(x)
        return self.tanh(self.fc_out(out))  # (batch, seq_len, output_dim)


class TemporalDiscriminator(nn.Module):
    """GRU-based discriminator: SCADA sequence → real/fake score."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])  # Use last hidden state


class TemporalEmbedder(nn.Module):
    """Embeds real sequences into latent space for supervised loss."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return torch.tanh(self.fc(out[:, -1, :]))


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

class TemporalGAN:
    def __init__(
        self,
        seq_len: int = 36,        # 6 hours at 10-min intervals
        feature_dim: int = 9,     # SCADA features
        latent_dim: int = 64,
        hidden_dim: int = 128,
        device: str = "auto",
    ):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

        self.G = TemporalGenerator(latent_dim, hidden_dim, feature_dim, seq_len).to(self.device)
        self.D = TemporalDiscriminator(feature_dim, hidden_dim).to(self.device)
        self.E = TemporalEmbedder(feature_dim, hidden_dim, latent_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        print(f"[TemporalGAN] Initialized | Device: {self.device}")
        print(f"  Generator params:     {sum(p.numel() for p in self.G.parameters()):,}")
        print(f"  Discriminator params: {sum(p.numel() for p in self.D.parameters()):,}")

    def _prepare_sequences(self, data: np.ndarray) -> TensorDataset:
        """Sliding window to create (batch, seq_len, features) tensor."""
        sequences = []
        for i in range(len(data) - self.seq_len):
            sequences.append(data[i: i + self.seq_len])
        X = torch.FloatTensor(np.array(sequences))
        return TensorDataset(X)

    def fit(self, data: np.ndarray, epochs: int = 100, batch_size: int = 64):
        """
        Train the Temporal GAN.

        Args:
            data: Normalized SCADA array of shape (timesteps, features)
            epochs: Training epochs
            batch_size: Mini-batch size
        """
        dataset = self._prepare_sequences(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        history = {"d_loss": [], "g_loss": []}

        for epoch in range(1, epochs + 1):
            d_losses, g_losses = [], []

            for (real_seqs,) in loader:
                real_seqs = real_seqs.to(self.device)
                bs = real_seqs.size(0)

                # ── Discriminator Step ──────────────────────
                self.opt_D.zero_grad()
                z = torch.randn(bs, self.latent_dim).to(self.device)
                fake_seqs = self.G(z).detach()

                real_labels = torch.ones(bs, 1).to(self.device)
                fake_labels = torch.zeros(bs, 1).to(self.device)

                d_real = self.criterion(self.D(real_seqs), real_labels)
                d_fake = self.criterion(self.D(fake_seqs), fake_labels)
                d_loss = (d_real + d_fake) / 2
                d_loss.backward()
                self.opt_D.step()

                # ── Generator Step ──────────────────────────
                self.opt_G.zero_grad()
                z = torch.randn(bs, self.latent_dim).to(self.device)
                fake_seqs = self.G(z)

                g_adv = self.criterion(self.D(fake_seqs), real_labels)

                # Supervised temporal consistency loss
                real_emb = self.E(real_seqs).detach()
                fake_emb = self.E(fake_seqs)
                g_sup = self.mse(fake_emb, real_emb)

                g_loss = g_adv + 10 * g_sup
                g_loss.backward()
                self.opt_G.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            avg_d = np.mean(d_losses)
            avg_g = np.mean(g_losses)
            history["d_loss"].append(avg_d)
            history["g_loss"].append(avg_g)

            if epoch % 10 == 0:
                print(f"  Epoch [{epoch:03d}/{epochs}] | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

        return history

    def generate(self, n_samples: int = 100) -> np.ndarray:
        """Generate synthetic SCADA sequences."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            synthetic = self.G(z).cpu().numpy()
        self.G.train()
        return synthetic  # (n_samples, seq_len, feature_dim)

    def generate_counterfactual(
        self, real_seq: np.ndarray, feature_idx: int, perturb_value: float
    ) -> np.ndarray:
        """
        Generate 'what-if' scenarios by perturbing a specific sensor.

        Args:
            real_seq: (seq_len, features) — base sequence
            feature_idx: Which feature to perturb
            perturb_value: New value to inject
        """
        counterfactual = real_seq.copy()
        counterfactual[:, feature_idx] = perturb_value
        return counterfactual

    def save(self, path: str = "temporal_gan.pt"):
        torch.save({
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "E": self.E.state_dict(),
        }, path)
        print(f"[TemporalGAN] Saved to {path}")

    def load(self, path: str = "temporal_gan.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.E.load_state_dict(ckpt["E"])
        print(f"[TemporalGAN] Loaded from {path}")


if __name__ == "__main__":
    # Quick smoke test with random data
    dummy_data = np.random.randn(500, 9).astype(np.float32)
    gan = TemporalGAN(seq_len=36, feature_dim=9)
    history = gan.fit(dummy_data, epochs=20, batch_size=32)
    synthetic = gan.generate(n_samples=10)
    print(f"\n[Test] Synthetic shape: {synthetic.shape}")
