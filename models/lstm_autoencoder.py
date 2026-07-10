import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run:  pip install torch")
    print("Install guide: https://pytorch.org/get-started/locally/")

import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "Week - 4" / "data" / "features.csv"
SAVE_DIR = BASE_DIR / "saved"
SAVE_DIR.mkdir(exist_ok=True)

SEQUENCE_LEN = 10
HIDDEN_SIZE = 64
LATENT_SIZE = 32
NUM_LAYERS = 2
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
THRESHOLD_PERCENTILE = 95


DROP_COLS = [
    "event_id",
    "user_id",
    "timestamp",
    "event_type",
    "ip_address",
    "country",
    "device",
    "method",
    "instrument",
    "kyc_change_type",
    "anomaly_type",
    "is_anomalous",
]


class LSTMEncoder(nn.Module):
    """
    Encodes a sequence of events into a fixed-size latent vector.
    Takes input of shape (batch, seq_len, n_features)
    Returns latent vector of shape (batch, latent_size)
    """

    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        latent = self.fc(last_hidden)
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.fc = nn.Linear(latent_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, latent):
        h = self.fc(latent)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        reconstructed = self.output_layer(out)
        return reconstructed


class LSTMAutoencoder(nn.Module):

    def __init__(self, input_size, hidden_size, latent_size, num_layers, seq_len):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_size, num_layers)
        self.decoder = LSTMDecoder(
            latent_size, hidden_size, input_size, num_layers, seq_len
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def load_and_prepare(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    y = df["is_anomalous"].values
    user = df["user_id"].values

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])

    print(f"  Shape: {df.shape}  |  Features: {X.shape[1]}")
    return X.values, y, user, list(X.columns)


def build_sequences(X, y, user_ids, seq_len):

    sequences = []
    labels = []

    for uid in np.unique(user_ids):
        mask = user_ids == uid
        X_user = X[mask]
        y_user = y[mask]

        for i in range(len(X_user) - seq_len + 1):
            seq = X_user[i : i + seq_len]
            seq_lbl = int(y_user[i : i + seq_len].any())
            sequences.append(seq)
            labels.append(seq_lbl)

    return np.array(sequences, dtype=np.float32), np.array(labels)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        reconstructed = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def compute_mse(model, sequences, device, batch_size=128):
    """Compute per-sequence MSE (reconstruction error)."""
    model.eval()
    all_mse = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[i : i + batch_size]).to(device)
            reconstructed = model(batch)
            mse = ((batch - reconstructed) ** 2).mean(dim=(1, 2))
            all_mse.extend(mse.cpu().numpy())
    return np.array(all_mse)


#  Main pipeline


def run():
    if not TORCH_AVAILABLE:
        print("Skipping LSTM training — PyTorch not available.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    X, y, user_ids, feature_names = load_and_prepare(DATA_PATH)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # build sequences
    print(f"\nBuilding sequences (seq_len={SEQUENCE_LEN})...")
    sequences, seq_labels = build_sequences(X_scaled, y, user_ids, SEQUENCE_LEN)
    print(f"  Total sequences  : {len(sequences)}")
    print(f"  Normal sequences : {(seq_labels == 0).sum()}")
    print(f"  Anomaly sequences: {(seq_labels == 1).sum()}")

    normal_seqs = sequences[seq_labels == 0]
    print(f"\nTraining on {len(normal_seqs)} normal sequences only...")

    train_tensor = torch.tensor(normal_seqs, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True
    )

    # model
    input_size = X.shape[1]
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        num_layers=NUM_LAYERS,
        seq_len=SEQUENCE_LEN,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # training loop
    print()
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  |  loss: {loss:.6f}")

    print("\nComputing reconstruction errors...")
    all_mse = compute_mse(model, sequences, device)

    normal_mse = all_mse[seq_labels == 0]
    threshold = float(np.percentile(normal_mse, THRESHOLD_PERCENTILE))
    print(f"  Threshold (p{THRESHOLD_PERCENTILE} of normal MSE): {threshold:.6f}")

    preds = (all_mse > threshold).astype(int)
    from sklearn.metrics import roc_auc_score, classification_report

    roc = roc_auc_score(seq_labels, all_mse)
    print(f"  ROC-AUC: {roc:.4f}")
    print()
    print(
        classification_report(seq_labels, preds, target_names=["normal", "anomalous"])
    )

    torch.save(model.state_dict(), SAVE_DIR / "lstm_autoencoder.pt")
    joblib.dump(scaler, SAVE_DIR / "lstm_scaler.pkl")
    joblib.dump(threshold, SAVE_DIR / "lstm_threshold.pkl")
    joblib.dump(
        {
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "latent_size": LATENT_SIZE,
            "num_layers": NUM_LAYERS,
            "seq_len": SEQUENCE_LEN,
            "feature_names": feature_names,
        },
        SAVE_DIR / "lstm_config.pkl",
    )

    print(f"✅ Saved to {SAVE_DIR}/")
    print(f"   lstm_autoencoder.pt")
    print(f"   lstm_scaler.pkl")
    print(f"   lstm_threshold.pkl")
    print(f"   lstm_config.pkl")


if __name__ == "__main__":
    run()
