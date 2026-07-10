# api/schemas.py
#
# Pydantic models for request and response validation.
# FastAPI uses these to automatically validate incoming data
# and generate the /docs page.

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request ───────────────────────────────────────────────────────────────────

class EventRequest(BaseModel):
    """
    A single event sent to /score.
    Contains all numeric features produced by feature_engineering.py.
    String columns like country, device, event_type are passed separately
    for display purposes but are not fed into the model.
    """

    # identifiers (not fed into model, used for response context)
    user_id:    str = Field(..., example="USER_0042")
    event_type: str = Field(..., example="withdrawal")

    # time features
    hour_of_day:  int   = Field(0,   ge=0, le=23)
    day_of_week:  int   = Field(0,   ge=0, le=6)
    is_weekend:   int   = Field(0,   ge=0, le=1)
    account_age_days: int = Field(0, ge=0)

    # login features
    login_success:    Optional[float] = Field(0.0)
    failed_attempts:  Optional[float] = Field(0.0)
    timezone_gap_hours: Optional[float] = Field(0.0)

    # trade features
    lot_size:                  Optional[float] = Field(0.0)
    trade_volume:              Optional[float] = Field(0.0)
    pnl:                       Optional[float] = Field(0.0)
    margin_used:               Optional[float] = Field(0.0)
    trade_duration_seconds:    Optional[float] = Field(0.0)
    trade_volume_vs_baseline:  Optional[float] = Field(0.0)
    is_night_trade:            Optional[float] = Field(0.0)

    # deposit / withdrawal features
    amount:                   Optional[float] = Field(0.0)
    is_immediate_withdrawal:  Optional[float] = Field(0.0)

    # session features
    session_duration_mins: Optional[float] = Field(0.0)
    page_clicks:           Optional[float] = Field(0.0)
    click_rate_per_min:    Optional[float] = Field(0.0)

    # time-delta features
    time_since_last_event_sec:   Optional[float] = Field(0.0)
    time_since_last_login_sec:   Optional[float] = Field(0.0)
    time_since_last_deposit_sec: Optional[float] = Field(0.0)

    # rolling trade features
    roll_5_trade_vol_mean:  Optional[float] = Field(0.0)
    roll_5_trade_vol_std:   Optional[float] = Field(0.0)
    roll_5_pnl_mean:        Optional[float] = Field(0.0)
    roll_5_margin_mean:     Optional[float] = Field(0.0)
    roll_10_trade_vol_mean: Optional[float] = Field(0.0)
    roll_10_trade_vol_std:  Optional[float] = Field(0.0)
    roll_10_pnl_mean:       Optional[float] = Field(0.0)
    roll_10_margin_mean:    Optional[float] = Field(0.0)

    # rolling session features
    roll_5_click_rate_mean:   Optional[float] = Field(0.0)
    roll_5_session_dur_mean:  Optional[float] = Field(0.0)
    roll_10_click_rate_mean:  Optional[float] = Field(0.0)
    roll_10_session_dur_mean: Optional[float] = Field(0.0)

    # burst count features
    burst_count_5min:  int = Field(0, ge=0)
    burst_count_30min: int = Field(0, ge=0)

    # login anomaly features
    unique_ips_last_10_logins:       Optional[float] = Field(0.0)
    unique_countries_last_10_logins: Optional[float] = Field(0.0)
    unique_devices_last_10_logins:   Optional[float] = Field(0.0)
    rolling_failed_attempts_5:       Optional[float] = Field(0.0)
    rolling_timezone_gap_5:          Optional[float] = Field(0.0)

    # financial ratio features
    roll_5_deposit_sum:             Optional[float] = Field(0.0)
    withdrawal_to_deposit_ratio:    Optional[float] = Field(0.0)
    rolling_immediate_withdrawal_5: Optional[float] = Field(0.0)

    # structurer features
    deposit_amount_rolling_std:  Optional[float] = Field(0.0)
    deposit_count_last_10:       Optional[float] = Field(0.0)
    deposit_amount_rolling_mean: Optional[float] = Field(0.0)

    # dormant withdrawer features
    withdrawal_amount_vs_account_age: Optional[float] = Field(0.0)
    is_large_withdrawal:              int = Field(0, ge=0, le=1)

    # z-score features
    trade_vol_zscore:            Optional[float] = Field(0.0)
    pnl_zscore:                  Optional[float] = Field(0.0)
    amount_zscore:               Optional[float] = Field(0.0)
    session_duration_zscore:     Optional[float] = Field(0.0)
    click_rate_zscore:           Optional[float] = Field(0.0)
    trade_vol_vs_baseline_zscore: Optional[float] = Field(0.0)

    class Config:
        # allows extra fields to be passed without raising an error
        # useful when consumer sends the full features.csv row
        extra = "ignore"


# ── Feature detail in response ────────────────────────────────────────────────

class FeatureDetail(BaseModel):
    """One feature's contribution to the anomaly score."""
    feature:      str
    raw_value:    float
    scaled_value: float


# ── Response ──────────────────────────────────────────────────────────────────

class ScoreResponse(BaseModel):
    """
    Response from /score endpoint.
    Contains the anomaly score, severity, human-readable verdict,
    plain-English reasons, and top contributing features.
    """
    user_id:       str
    event_type:    str
    anomaly_score: float
    is_anomaly:    bool
    severity:      str          # LOW / MEDIUM / HIGH / CRITICAL
    verdict:       str          # emoji + label for quick reading
    reasons:       List[str]    # 2-3 plain English explanations
    top_features:  List[FeatureDetail]  # top contributing features


# ── Health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
