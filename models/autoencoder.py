import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from sklearn.metrics import roc_auc_score

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
AE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "autoencoder.pt")


class FraudAutoencoder(nn.Module):
    """29 → 16 → 8 (bottleneck) → 16 → 29"""
    def __init__(self, input_dim: int = 29):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def reconstruction_error(model: FraudAutoencoder, X: np.ndarray,
                         device: torch.device) -> np.ndarray:
    """Compute per-sample MSE between input and reconstruction."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        recon = model(X_t)
        mse = ((X_t - recon) ** 2).mean(dim=1).cpu().numpy()
    return mse


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)


def train_autoencoder(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      epochs: int = 50, lr: float = 1e-3, batch_size: int = 512):

    # Use Apple Silicon GPU if available, then NVIDIA, then CPU
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # Train on normals only — the model learns what "normal" looks like
    X_train_normal = X_train[y_train == 0]
    dataset = TensorDataset(torch.FloatTensor(X_train_normal))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FraudAutoencoder(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name="autoencoder"):
        # Log hyperparams so you can compare across experiments
        mlflow.log_params({
            "model": "Autoencoder",
            "architecture": "29-16-8-16-29",
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "device": str(device),
            "train_normal_samples": len(X_train_normal),
        })

        # === Training loop ===
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for (batch,) in loader:
                batch = batch.to(device)

                # Forward: try to reconstruct the input
                recon = model(batch)

                # Loss: how different is the reconstruction from the original?
                loss = criterion(recon, batch)

                # Backward: compute gradients
                optimizer.zero_grad()
                loss.backward()

                # Update weights
                optimizer.step()

                epoch_loss += loss.item() * len(batch)

            avg_loss = epoch_loss / len(X_train_normal)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.6f}")

        # === Evaluate on test set ===
        test_errors = reconstruction_error(model, X_test, device)
        test_scores = normalize_scores(test_errors)
        auc = roc_auc_score(y_test, test_scores)

        mlflow.log_metric("roc_auc", auc)
        print(f"Autoencoder ROC-AUC: {auc:.4f}")

        # Save model weights + architecture info
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
        }, AE_MODEL_PATH)
        mlflow.log_artifact(AE_MODEL_PATH)

    return model, auc, device
