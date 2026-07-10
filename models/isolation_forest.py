import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "Week - 4" / "data" / "features.csv"
SAVE_DIR = BASE_DIR / "saved"
SAVE_DIR.mkdir(exist_ok=True)


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


def load_features(path):
    print(f"Loading features from {path}...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    return df


def prepare_features(df):
    """
    Drop non-numeric and identifier columns.
    Keep only the engineered numeric features for training.
    """
    y = df["is_anomalous"].values

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)

    X = X.select_dtypes(include=[np.number])

    print(f"  Features used for training: {X.shape[1]}")
    print(f"  Anomalous events: {y.sum()} / {len(y)}")
    return X, y


def train(X, contamination=0.15):
    """
    Train Isolation Forest.

    contamination=0.15 because ~19% of our events are anomalous.
    Setting it slightly lower than actual rate to reduce false positives.
    Higher contamination = model flags more events as anomalous.
    n_estimators=200 gives more stable scores than the default 100.
    random_state=42 for reproducibility.
    """
    print("\nTraining Isolation Forest...")
    print(f"  contamination = {contamination}")
    print(f"  n_estimators  = 200")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
    )
    model.fit(X_scaled)

    return model, scaler, X_scaled


def evaluate(model, scaler, X_scaled, y, X):
    """
    Evaluate the model and print results.
    Also saves a confusion matrix plot.
    """
    print("\nEvaluating...")

    preds_raw = model.predict(X_scaled)
    preds = np.where(preds_raw == -1, 1, 0)

    scores = -model.decision_function(X_scaled)

    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(y, preds, target_names=["normal", "anomalous"]))

    roc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)
    print(f"ROC-AUC           : {roc:.4f}")
    print(f"Average Precision  : {ap:.4f}")

    cm = confusion_matrix(y, preds)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay(cm, display_labels=["normal", "anomalous"]).plot(
        ax=axes[0], colorbar=False
    )
    axes[0].set_title("Confusion Matrix")

    axes[1].hist(scores[y == 0], bins=60, alpha=0.6, label="Normal", color="steelblue")
    axes[1].hist(scores[y == 1], bins=60, alpha=0.6, label="Anomalous", color="tomato")
    axes[1].set_xlabel("Anomaly Score (higher = more suspicious)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Score Distribution: Normal vs Anomalous")
    axes[1].legend()

    plt.tight_layout()
    plot_path = SAVE_DIR / "if_evaluation.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nEvaluation plot saved to {plot_path}")
    plt.close()

    print("\n── Score by Anomaly Type ─────────────────────────────")

    return scores, preds, roc, ap


def save_artifacts(model, scaler, scores, y):
    """
    Save model, scaler, and threshold.
    Threshold is set at the 80th percentile of normal event scores.
    Anything above this is flagged as anomalous.
    """

    threshold = float(np.percentile(scores[y == 0], 99))

    joblib.dump(model, SAVE_DIR / "isolation_forest.pkl")
    joblib.dump(scaler, SAVE_DIR / "if_scaler.pkl")
    joblib.dump(threshold, SAVE_DIR / "if_threshold.pkl")

    print(f"\n── Saved Artifacts ───────────────────────────────────")
    print(f"  isolation_forest.pkl  →  {SAVE_DIR / 'isolation_forest.pkl'}")
    print(f"  if_scaler.pkl         →  {SAVE_DIR / 'if_scaler.pkl'}")
    print(f"  if_threshold.pkl      →  {SAVE_DIR / 'if_threshold.pkl'}")
    print(f"  Threshold value       :  {threshold:.4f}")
    print(f"  (events with score > {threshold:.2f} will be flagged)")


def score_breakdown(df, scores):
    """Print mean anomaly score per anomaly type — sanity check."""
    df = df.copy()
    df["_score"] = scores
    print("\n── Mean Score per Anomaly Type ───────────────────────")
    breakdown = df.groupby("anomaly_type")["_score"].mean().sort_values(ascending=False)
    print(breakdown.to_string())
    print()
    print("  (anomalous types should have higher scores than 'none')")


def run():
    df = load_features(DATA_PATH)
    X, y = prepare_features(df)
    model, scaler, X_scaled = train(X, contamination=0.15)
    scores, preds, roc, ap = evaluate(model, scaler, X_scaled, y, X)
    score_breakdown(df, scores)
    save_artifacts(model, scaler, scores, y)

    print(f"\n✅ Done — ROC-AUC: {roc:.4f}  |  Avg Precision: {ap:.4f}")
    print("   Run the API next: uvicorn api.main:app --reload")


if __name__ == "__main__":
    run()
