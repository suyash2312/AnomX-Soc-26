import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "saved"

FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "login_success",
    "failed_attempts",
    "timezone_gap_hours",
    "lot_size",
    "trade_volume",
    "pnl",
    "margin_used",
    "trade_duration_seconds",
    "trade_volume_vs_baseline",
    "is_night_trade",
    "amount",
    "is_immediate_withdrawal",
    "session_duration_mins",
    "page_clicks",
    "click_rate_per_min",
    "account_age_days",
    "time_since_last_event_sec",
    "time_since_last_login_sec",
    "time_since_last_deposit_sec",
    "roll_5_trade_vol_mean",
    "roll_5_trade_vol_std",
    "roll_5_pnl_mean",
    "roll_5_margin_mean",
    "roll_10_trade_vol_mean",
    "roll_10_trade_vol_std",
    "roll_10_pnl_mean",
    "roll_10_margin_mean",
    "roll_5_click_rate_mean",
    "roll_5_session_dur_mean",
    "roll_10_click_rate_mean",
    "roll_10_session_dur_mean",
    "burst_count_5min",
    "burst_count_30min",
    "unique_ips_last_10_logins",
    "unique_countries_last_10_logins",
    "unique_devices_last_10_logins",
    "rolling_failed_attempts_5",
    "rolling_timezone_gap_5",
    "roll_5_deposit_sum",
    "withdrawal_to_deposit_ratio",
    "rolling_immediate_withdrawal_5",
    "deposit_amount_rolling_std",
    "deposit_count_last_10",
    "deposit_amount_rolling_mean",
    "withdrawal_amount_vs_account_age",
    "is_large_withdrawal",
    "trade_vol_zscore",
    "pnl_zscore",
    "amount_zscore",
    "session_duration_zscore",
    "click_rate_zscore",
    "trade_vol_vs_baseline_zscore",
]

FEATURE_DESCRIPTIONS = {
    "amount": "Unusual transaction amount",
    "withdrawal_to_deposit_ratio": "Abnormal withdrawal behaviour",
    "failed_attempts": "Multiple failed login attempts",
    "rolling_failed_attempts_5": "Failed login attempts building up",
    "burst_count_5min": "High activity burst in short window",
    "burst_count_30min": "Sustained high activity detected",
    "unique_countries_last_10_logins": "Logins from multiple countries",
    "unique_ips_last_10_logins": "Multiple IPs used recently",
    "unique_devices_last_10_logins": "Multiple devices used recently",
    "rolling_timezone_gap_5": "Logins from unusual timezones",
    "timezone_gap_hours": "Login from unexpected timezone",
    "trade_volume": "Unusually large trade volume",
    "trade_vol_zscore": "Trade volume far above user's normal",
    "trade_volume_vs_baseline": "Trade volume vs user baseline is extreme",
    "roll_5_trade_vol_mean": "Recent trades much larger than usual",
    "pnl": "Abnormal profit/loss pattern",
    "pnl_zscore": "PnL far outside user's normal range",
    "roll_5_pnl_mean": "Consistently unusual profit pattern",
    "click_rate_per_min": "Inhuman click rate detected",
    "click_rate_zscore": "Click rate far above normal",
    "roll_5_click_rate_mean": "Sustained high click rate",
    "session_duration_mins": "Unusual session duration",
    "is_immediate_withdrawal": "Withdrawal immediately after deposit",
    "rolling_immediate_withdrawal_5": "Repeated deposit-withdrawal cycling",
    "deposit_amount_rolling_std": "Deposits suspiciously similar in size",
    "deposit_count_last_10": "Unusually high number of deposits",
    "is_large_withdrawal": "Large withdrawal detected",
    "withdrawal_amount_vs_account_age": "Large withdrawal on dormant account",
    "amount_zscore": "Transaction amount far outside normal",
    "is_night_trade": "Trading activity at unusual hours",
    "hour_of_day": "Activity at unusual time of day",
    "margin_used": "Unusually high margin usage",
    "lot_size": "Abnormal lot size",
}

SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.06,
    "HIGH": 0.03,
    "MEDIUM": 0.01,
    "LOW": float("-inf"),
}


class ForexGuardScorer:

    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None
        self.loaded = False

    def load(self):

        try:
            self.model = joblib.load(MODELS_DIR / "isolation_forest.pkl")
            self.scaler = joblib.load(MODELS_DIR / "if_scaler.pkl")
            self.threshold = joblib.load(MODELS_DIR / "if_threshold.pkl")
            self.loaded = True
            print(f"[ForexGuardScorer] Model loaded from {MODELS_DIR}")
        except FileNotFoundError as e:
            print(f"[ForexGuardScorer] ERROR: Could not load model — {e}")
            print("  Run  python models/isolation_forest.py  first to train the model.")
            self.loaded = False

    def _build_feature_vector(self, event: dict):
        import pandas as pd

        row = {col: float(event.get(col, 0.0)) for col in FEATURE_COLS}
        return pd.DataFrame([row])

    def _get_severity(self, score: float) -> str:
        for level, threshold in SEVERITY_THRESHOLDS.items():
            if score >= threshold:
                return level
        return "LOW"

    def _get_top_features(
        self, raw_vector: np.ndarray, scaled_vector: np.ndarray, top_n: int = 5
    ):
        raw = raw_vector.values.flatten()
        scaled = (
            scaled_vector.flatten()
            if hasattr(scaled_vector, "flatten")
            else scaled_vector.values.flatten()
        )

        indices = np.argsort(np.abs(scaled))[::-1][:top_n]

        top = []
        for idx in indices:
            top.append(
                {
                    "feature": FEATURE_COLS[idx],
                    "raw_value": round(float(raw[idx]), 4),
                    "scaled_value": round(float(scaled[idx]), 4),
                }
            )
        return top

    def _get_reasons(self, top_features: list) -> list:
        reasons = []
        for feat in top_features:
            name = feat["feature"]
            desc = FEATURE_DESCRIPTIONS.get(name)
            if desc and desc not in reasons:
                reasons.append(desc)
            if len(reasons) == 3:
                break

        if not reasons:
            reasons = ["Unusual activity pattern detected"]

        return reasons

    def score(self, event: dict) -> dict:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call scorer.load() first.")

        raw_vector = self._build_feature_vector(event)
        scaled_vector = self.scaler.transform(raw_vector)

        raw_score = float(-self.model.decision_function(scaled_vector)[0])
        is_anomaly = raw_score > self.threshold

        severity = self._get_severity(raw_score) if is_anomaly else "NONE"
        verdict = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"

        top_features = self._get_top_features(raw_vector, scaled_vector)
        reasons = self._get_reasons(top_features) if is_anomaly else []

        return {
            "user_id": event.get("user_id", "unknown"),
            "event_type": event.get("event_type", "unknown"),
            "anomaly_score": round(raw_score, 6),
            "is_anomaly": bool(is_anomaly),
            "severity": severity,
            "verdict": verdict,
            "reasons": reasons,
            "top_features": top_features,
        }
