import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_PATH = BASE_DIR / "data" / "events.csv"
OUT_PATH = BASE_DIR / "data" / "features.csv"


ROLLING_WINDOWS = [5, 10]


def _safe_zscore(series: pd.Series) -> pd.Series:

    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _time_since_etype(group: pd.DataFrame, etype: str) -> pd.Series:

    mask = group["event_type"] == etype
    last_ts = group["timestamp"].where(mask).ffill().shift(1)
    delta = (group["timestamp"] - last_ts).dt.total_seconds().fillna(0).clip(lower=0)
    return delta


def _burst_count(group: pd.DataFrame, window_min: int) -> pd.Series:

    ts_list = group["timestamp"].tolist()
    counts = []
    for i, t in enumerate(ts_list):
        cutoff = t - pd.Timedelta(minutes=window_min)
        counts.append(sum(1 for x in ts_list[: i + 1] if x >= cutoff))
    return pd.Series(counts, index=group.index)


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [1/9] Time-delta features...")

    df["time_since_last_event_sec"] = (
        df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    )

    df["time_since_last_login_sec"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: _time_since_etype(g, "login")
    )

    df["time_since_last_deposit_sec"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: _time_since_etype(g, "deposit")
    )

    return df


def build_rolling_trade_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [2/9] Rolling trade features...")

    trade_df = df[df["event_type"] == "trade"].copy()

    for w in ROLLING_WINDOWS:
        for col, agg in [
            ("trade_volume", "mean"),
            ("trade_volume", "std"),
            ("pnl", "mean"),
            ("margin_used", "mean"),
        ]:

            short = col.replace("_volume", "_vol").replace("_used", "")
            feat = f"roll_{w}_{short}_{agg}"
            rolled = (
                trade_df.groupby("user_id")[col]
                .rolling(w, min_periods=1)
                .agg(agg)
                .reset_index(level=0, drop=True)
            )
            df[feat] = rolled

    return df


def build_rolling_session_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [3/9] Rolling session features...")

    sess_df = df[df["event_type"] == "session"].copy()

    for w in ROLLING_WINDOWS:
        for col, feat in [
            ("click_rate_per_min", f"roll_{w}_click_rate_mean"),
            ("session_duration_mins", f"roll_{w}_session_dur_mean"),
        ]:
            rolled = (
                sess_df.groupby("user_id")[col]
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[feat] = rolled

    return df


def build_burst_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [4/9] Burst count features...")

    df["burst_count_5min"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: _burst_count(g, 5)
    )
    df["burst_count_30min"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: _burst_count(g, 30)
    )
    return df


def build_login_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [5/9] Login anomaly features...")

    login_df = df[df["event_type"] == "login"].copy()

    for col, feat in [
        ("ip_address", "unique_ips_last_10_logins"),
        ("country", "unique_countries_last_10_logins"),
        ("device", "unique_devices_last_10_logins"),
    ]:
        login_df[col + "_enc"] = login_df.groupby("user_id")[col].transform(
            lambda s: pd.factorize(s)[0]
        )
        rolled = (
            login_df.groupby("user_id")[col + "_enc"]
            .rolling(10, min_periods=1)
            .apply(lambda x: len(set(x)), raw=False)
            .reset_index(level=0, drop=True)
        )
        df[feat] = rolled

    df["rolling_failed_attempts_5"] = (
        login_df.groupby("user_id")["failed_attempts"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["rolling_timezone_gap_5"] = (
        login_df.groupby("user_id")["timezone_gap_hours"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def build_financial_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [6/9] Financial ratio features...")

    fin_df = df[df["event_type"].isin(["deposit", "withdrawal"])].copy()

    dep_roll = (
        fin_df[fin_df["event_type"] == "deposit"]
        .groupby("user_id")["amount"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .rename("roll_5_deposit_sum")
    )
    fin_df = fin_df.join(dep_roll)
    fin_df["roll_5_deposit_sum"] = (
        fin_df.groupby("user_id")["roll_5_deposit_sum"].ffill().fillna(0)
    )

    fin_df["withdrawal_to_deposit_ratio"] = np.where(
        fin_df["event_type"] == "withdrawal",
        (fin_df["amount"] / (fin_df["roll_5_deposit_sum"] + 1e-9)).clip(upper=10),
        0.0,
    )

    imm_roll = (
        fin_df.groupby("user_id")["is_immediate_withdrawal"]
        .rolling(5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .rename("rolling_immediate_withdrawal_5")
    )
    fin_df = fin_df.join(imm_roll)

    df["roll_5_deposit_sum"] = fin_df["roll_5_deposit_sum"]
    df["withdrawal_to_deposit_ratio"] = fin_df["withdrawal_to_deposit_ratio"]
    df["rolling_immediate_withdrawal_5"] = fin_df["rolling_immediate_withdrawal_5"]

    return df


def build_structurer_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [7/9] Structurer features...")

    dep_df = df[df["event_type"] == "deposit"].copy()

    df["deposit_amount_rolling_std"] = (
        dep_df.groupby("user_id")["amount"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df["deposit_count_last_10"] = (
        dep_df.groupby("user_id")["amount"]
        .rolling(10, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
    )

    df["deposit_amount_rolling_mean"] = (
        dep_df.groupby("user_id")["amount"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def build_dormant_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [8/9] Dormant withdrawer features...")

    df["withdrawal_amount_vs_account_age"] = np.where(
        df["event_type"] == "withdrawal",
        df["amount"] / (df["account_age_days"] + 1),
        0.0,
    )

    df["is_large_withdrawal"] = np.where(
        (df["event_type"] == "withdrawal") & (df["amount"] > 5000), 1, 0
    )

    return df


def build_zscore_features(df: pd.DataFrame) -> pd.DataFrame:

    print("  [9/9] Z-score features...")

    for col, name in [
        ("trade_volume", "trade_vol_zscore"),
        ("pnl", "pnl_zscore"),
        ("amount", "amount_zscore"),
        ("session_duration_mins", "session_duration_zscore"),
        ("click_rate_per_min", "click_rate_zscore"),
        ("trade_volume_vs_baseline", "trade_vol_vs_baseline_zscore"),
    ]:
        df[name] = df.groupby("user_id")[col].transform(_safe_zscore)

    return df


def run_feature_pipeline():
    print(f"Loading raw data from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    orig_cols = set(df.columns)
    print(f"Raw shape: {df.shape}\n")
    print("Building features...")

    df = build_time_features(df)
    df = build_rolling_trade_features(df)
    df = build_rolling_session_features(df)
    df = build_burst_features(df)
    df = build_login_features(df)
    df = build_financial_features(df)
    df = build_structurer_features(df)
    df = build_dormant_features(df)
    df = build_zscore_features(df)

    new_cols = [c for c in df.columns if c not in orig_cols]
    df[new_cols] = df[new_cols].fillna(0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"✅ Features saved to : {OUT_PATH}")
    print(f"   Original columns  : {len(orig_cols)}")
    print(f"   New columns added : {len(new_cols)}")
    print(f"   Total columns     : {len(df.columns)}")
    print(f"\nNew features:")
    for c in new_cols:
        print(f"  + {c}")
    print(f"{'='*55}")
    return df


if __name__ == "__main__":
    run_feature_pipeline()
