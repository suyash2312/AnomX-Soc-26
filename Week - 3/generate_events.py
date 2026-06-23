import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# import yaml
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_USERS = 200
N_EVENTS = 5000
ANOMALY_FRACTION = 0.15
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 3, 31)

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "USDCHF", "AUDUSD"]
COUNTRIES = ["IN", "US", "UK", "SG", "AE", "NG", "RU", "CN", "DE", "BR"]
DEVICES = ["chrome_win", "safari_mac", "android_app", "ios_app", "firefox_linux"]
EVENT_TYPES = ["login", "trade", "deposit", "withdrawal", "session", "kyc_change"]
EVENT_WEIGHTS = [0.20, 0.33, 0.17, 0.10, 0.15, 0.05]
METHODS = ["card", "bank", "crypto"]

COUNTRY_TIMEZONES = {
    "IN": 5.5,
    "US": -5,
    "UK": 0,
    "SG": 8,
    "AE": 4,
    "NG": 1,
    "RU": 3,
    "CN": 8,
    "DE": 1,
    "BR": -3,
}

ANOMALY_TYPES = [
    "brute_forcer",
    "ip_hopper",
    "wash_trader",
    "structurer",
    "bot_trader",
    "dormant_withdrawer",
    "deposit_withdrawal_cycler",
]

# Helper Functions


def random_timestamp(start=START_DATE, end=END_DATE):
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def random_ip():
    return (
        f"192.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    )


def event_id():
    """Generate a unique event ID string."""
    return f"EVT_{random.randint(100000, 999999)}"


def timezone_offset(country):
    return COUNTRY_TIMEZONES.get(country, 0)


# User profile builder


def build_user_profiles():
    profiles = {}
    n_anomalous = int(N_USERS * ANOMALY_FRACTION)

    for i in range(N_USERS):
        user_id = f"USER_{i:04d}"
        is_anomalous = i < n_anomalous
        anomaly_type = ANOMALY_TYPES[i % len(ANOMALY_TYPES)] if is_anomalous else None

        home_country = random.choice(COUNTRIES)

        profiles[user_id] = {
            "home_ip": f"10.{random.randint(0,255)}.{random.randint(0,255)}.1",
            "home_country": home_country,
            "home_timezone_offset": timezone_offset(home_country),
            "preferred_device": random.choice(DEVICES),
            "instruments": random.sample(INSTRUMENTS, k=random.randint(1, 3)),
            "typical_trade_vol": round(np.random.uniform(1000, 20000), 2),
            "typical_deposit": round(np.random.uniform(200, 3000), 2),
            "account_created": random_timestamp(
                START_DATE - timedelta(days=365), START_DATE
            ),
            "is_anomalous": is_anomalous,
            "anomaly_type": anomaly_type,
        }
    return profiles


# ── Base event skeleton ───────────────────────────────────────────────────────


def base_event(user_id, ts):
    return {
        "event_id": event_id(),
        "user_id": user_id,
        "event_type": None,
        "timestamp": ts,
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),
        "is_anomalous": 0,
        "anomaly_type": "none",
        # ── Login fields ──────────────────────────────────────────────────────
        "ip_address": None,
        "country": None,
        "device": None,
        "login_success": None,
        "failed_attempts": None,
        "timezone_gap_hours": None,
        # ── Trade fields ──────────────────────────────────────────────────────
        "instrument": None,
        "lot_size": None,
        "trade_volume": None,
        "pnl": None,
        "margin_used": None,
        "trade_duration_seconds": None,
        "trade_volume_vs_baseline": None,  # actual_vol / user's typical_vol; >5x = red flag
        "is_night_trade": None,  # 1 if trade placed between midnight and 5 AM local
        # ── Deposit / withdrawal fields ───────────────────────────────────────
        "amount": None,
        "method": None,
        "is_immediate_withdrawal": None,  # 1 if withdrawal follows deposit within hours
        # ── Session fields ────────────────────────────────────────────────────
        "session_duration_mins": None,
        "page_clicks": None,
        "click_rate_per_min": None,  # clicks/min; >200 strongly suggests a bot
        # ── KYC fields ───────────────────────────────────────────────────────
        "kyc_change_type": None,
        # ── Account context ───────────────────────────────────────────────────
        "account_age_days": None,
    }


def generate_normal_event(user_id, profile, ts=None):
    """
    Generate one random normal event for a user.

    Event type is sampled from EVENT_TYPES using EVENT_WEIGHTS so the
    distribution matches realistic platform usage (trades most frequent,
    kyc_change rare). All values stay close to the user's baseline profile.
    """
    if ts is None:
        ts = random_timestamp()

    event_type = random.choices(EVENT_TYPES, weights=EVENT_WEIGHTS)[0]
    e = base_event(user_id, ts)
    e["event_type"] = event_type
    e["account_age_days"] = (ts - profile["account_created"]).days

    if event_type == "login":
        country = profile["home_country"]
        e.update(
            {
                "ip_address": profile["home_ip"],
                "country": country,
                "device": profile["preferred_device"],
                "login_success": 1,
                "failed_attempts": int(np.random.choice([0, 1], p=[0.95, 0.05])),
                # Normal login from home country → timezone gap is 0
                "timezone_gap_hours": abs(
                    timezone_offset(country) - profile["home_timezone_offset"]
                ),
            }
        )

    elif event_type == "trade":
        vol = profile["typical_trade_vol"]
        actual_vol = round(np.random.normal(vol, vol * 0.15), 2)
        local_hour = (ts.hour + profile["home_timezone_offset"]) % 24
        e.update(
            {
                "instrument": random.choice(profile["instruments"]),
                "lot_size": round(np.random.uniform(0.01, 2.0), 2),
                "trade_volume": actual_vol,
                "pnl": round(np.random.normal(0, 150), 2),
                "margin_used": round(np.random.uniform(100, 5000), 2),
                "trade_duration_seconds": int(np.random.uniform(60, 3600)),
                "trade_volume_vs_baseline": round(actual_vol / vol, 3),
                "is_night_trade": int(0 <= local_hour <= 5),  # local midnight–5 AM
            }
        )

    elif event_type == "deposit":
        e.update(
            {
                "amount": round(np.random.normal(profile["typical_deposit"], 150), 2),
                "method": random.choice(METHODS),
                "is_immediate_withdrawal": 0,
            }
        )

    elif event_type == "withdrawal":
        e.update(
            {
                "amount": round(
                    np.random.uniform(50, profile["typical_deposit"] * 0.7), 2
                ),
                "method": random.choice(["bank", "crypto"]),
                "is_immediate_withdrawal": 0,
            }
        )

    elif event_type == "session":
        duration = round(np.random.uniform(5, 60), 1)
        clicks = int(np.random.uniform(5, 80))
        e.update(
            {
                "device": profile["preferred_device"],
                "session_duration_mins": duration,
                "page_clicks": clicks,
                "click_rate_per_min": round(clicks / max(duration, 0.1), 2),
            }
        )

    elif event_type == "kyc_change":
        e.update(
            {
                "kyc_change_type": random.choice(
                    ["address_update", "phone_update", "email_update"]
                ),
            }
        )

    return e


def events_ip_hopper(user_id, profile):
    """
    Rapid logins from many different countries within minutes of each other.
    Key signals: high timezone_gap_hours, new IP + country on every login,
    impossible travel speed between login locations.
    """
    events = []
    base_ts = random_timestamp()
    n = random.randint(6, 10)
    prev_country = profile["home_country"]

    for i in range(n):
        ts = base_ts + timedelta(minutes=random.randint(1, 10))
        country = random.choice(COUNTRIES)
        tz_gap = abs(timezone_offset(country) - timezone_offset(prev_country))
        e = base_event(user_id, ts)
        e.update(
            {
                "event_type": "login",
                "ip_address": random_ip(),
                "country": country,
                "device": random.choice(DEVICES),
                "login_success": 1,
                "failed_attempts": 0,
                "timezone_gap_hours": tz_gap,  # large gap = geographically impossible travel
                "is_anomalous": 1,
                "anomaly_type": "ip_hopper",
                "account_age_days": (ts - profile["account_created"]).days,
            }
        )
        prev_country = country
        events.append(e)

    return events


def events_wash_trader(user_id, profile):
    """
    Trade volume 10-20x the user's baseline, always profitable, very fast execution.
    Key signals: trade_volume_vs_baseline >> 1, consistently positive pnl,
    trade_duration_seconds very low (sub-30s), single instrument.
    """
    events = []
    base_ts = random_timestamp()
    vol = profile["typical_trade_vol"]
    n = random.randint(8, 12)

    for i in range(n):
        ts = base_ts + timedelta(minutes=i * 3)
        actual_vol = round(vol * random.uniform(10, 20), 2)
        local_hour = (ts.hour + profile["home_timezone_offset"]) % 24
        e = base_event(user_id, ts)
        e.update(
            {
                "event_type": "trade",
                "instrument": INSTRUMENTS[0],  # always the same instrument
                "lot_size": round(np.random.uniform(20, 50), 2),
                "trade_volume": actual_vol,
                "pnl": round(np.random.uniform(500, 5000), 2),  # always profit
                "margin_used": round(np.random.uniform(50000, 200000), 2),
                "trade_duration_seconds": int(
                    np.random.uniform(1, 30)
                ),  # suspiciously fast
                "trade_volume_vs_baseline": round(actual_vol / vol, 3),
                "is_night_trade": int(0 <= local_hour <= 5),
                "is_anomalous": 1,
                "anomaly_type": "wash_trader",
                "account_age_days": (ts - profile["account_created"]).days,
            }
        )
        events.append(e)

    return events


def events_deposit_withdrawal_cycler(user_id, profile):
    """
    Deposit followed by withdrawal of nearly the same amount within hours.
    Classic money-laundering layering pattern — funds enter and exit rapidly.
    Key signals: is_immediate_withdrawal=1, deposit and withdrawal amounts close,
    method is crypto (harder to trace).
    """
    events = []
    base_ts = random_timestamp()
    n = random.randint(3, 5)  # repeat the cycle multiple times

    for i in range(n):
        ts_dep = base_ts + timedelta(hours=i * 10)
        dep_amount = round(np.random.uniform(5000, 15000), 2)

        e_dep = base_event(user_id, ts_dep)
        e_dep.update(
            {
                "event_type": "deposit",
                "amount": dep_amount,
                "method": "crypto",
                "is_immediate_withdrawal": 1,
                "is_anomalous": 1,
                "anomaly_type": "deposit_withdrawal_cycler",
                "account_age_days": (ts_dep - profile["account_created"]).days,
            }
        )
        events.append(e_dep)

        # Withdrawal happens 1–3 hours after deposit, taking 85-98% of the deposit
        ts_wit = ts_dep + timedelta(hours=random.randint(1, 3))
        e_wit = base_event(user_id, ts_wit)
        e_wit.update(
            {
                "event_type": "withdrawal",
                "amount": round(dep_amount * random.uniform(0.85, 0.98), 2),
                "method": "crypto",
                "is_immediate_withdrawal": 1,
                "is_anomalous": 1,
                "anomaly_type": "deposit_withdrawal_cycler",
                "account_age_days": (ts_wit - profile["account_created"]).days,
            }
        )
        events.append(e_wit)

    return events


def events_bot_trader(user_id, profile):
    """
    Sessions with inhuman click rates (300-800 clicks/min) at odd hours (2-4 AM).
    Key signals: click_rate_per_min far exceeds human capability (~5-10/min),
    very short session_duration_mins, always active at night.
    """
    events = []
    base_ts = random_timestamp()
    base_ts = base_ts.replace(hour=random.randint(2, 4), minute=0)
    n = random.randint(6, 10)

    for i in range(n):
        ts = base_ts + timedelta(minutes=i * 2)
        duration = round(np.random.uniform(0.2, 1.5), 1)
        clicks = int(np.random.uniform(300, 800))
        e = base_event(user_id, ts)
        e.update(
            {
                "event_type": "session",
                "device": "chrome_win",
                "session_duration_mins": duration,
                "page_clicks": clicks,
                "click_rate_per_min": round(clicks / max(duration, 0.1), 2),
                "is_anomalous": 1,
                "anomaly_type": "bot_trader",
                "account_age_days": (ts - profile["account_created"]).days,
            }
        )
        events.append(e)

    return events


def events_structurer(user_id, profile):
    """
    Many deposits just under $1000 — classic smurfing / structuring pattern
    used to avoid AML (anti-money-laundering) reporting thresholds.
    Key signal: many deposits with amount in 490-999 range over short time.
    """
    events = []
    base_ts = random_timestamp()
    n = random.randint(10, 15)

    for i in range(n):
        ts = base_ts + timedelta(hours=i * 2)
        e = base_event(user_id, ts)
        e.update(
            {
                "event_type": "deposit",
                "amount": round(np.random.uniform(490, 999), 2),
                "method": random.choice(["card", "crypto"]),
                "is_immediate_withdrawal": 0,
                "is_anomalous": 1,
                "anomaly_type": "structurer",
                "account_age_days": (ts - profile["account_created"]).days,
            }
        )
        events.append(e)

    return events


def events_brute_forcer(user_id, profile):
    """
    Several failed logins in rapid succession (every 30s), then one success.
    Key signals: escalating failed_attempts, multiple countries, short time window.
    """
    events = []
    base_ts = random_timestamp()
    n_fails = random.randint(4, 8)

    for i in range(n_fails):
        ts = base_ts + timedelta(seconds=i * 30)
        country = random.choice(COUNTRIES)
        e = base_event(user_id, ts)
        e.update(
            {
                "event_type": "login",
                "ip_address": random_ip(),
                "country": country,
                "device": random.choice(DEVICES),
                "login_success": 0,
                "failed_attempts": i + 1,
                "timezone_gap_hours": abs(
                    timezone_offset(country) - profile["home_timezone_offset"]
                ),
                "is_anomalous": 1,
                "anomaly_type": "brute_forcer",
                "account_age_days": (ts - profile["account_created"]).days,
            }
        )
        events.append(e)

    # Final successful login right after all the failures
    ts_ok = base_ts + timedelta(seconds=n_fails * 30)
    country = random.choice(COUNTRIES)
    e_ok = base_event(user_id, ts_ok)
    e_ok.update(
        {
            "event_type": "login",
            "ip_address": random_ip(),
            "country": country,
            "device": random.choice(DEVICES),
            "login_success": 1,
            "failed_attempts": n_fails,
            "timezone_gap_hours": abs(
                timezone_offset(country) - profile["home_timezone_offset"]
            ),
            "is_anomalous": 1,
            "anomaly_type": "brute_forcer",
            "account_age_days": (ts_ok - profile["account_created"]).days,
        }
    )
    events.append(e_ok)
    return events


def events_dormant_withdrawer(user_id, profile):
    """
    Account stays quiet for a long time then suddenly makes large withdrawals.
    Key signal: account_age_days very high, no prior activity, large amount.
    Often indicates a compromised dormant account.
    """
    events = []
    # Force events near the end of the dataset window (after long dormancy)
    base_ts = END_DATE - timedelta(days=random.randint(1, 5))
    n = random.randint(2, 4)

    for i in range(n):
        ts = base_ts + timedelta(hours=i)
        e = base_event(user_id, ts)
        e.update(
            {
                "event_type": "withdrawal",
                "amount": round(np.random.uniform(10000, 40000), 2),
                "method": "crypto",
                "is_immediate_withdrawal": 0,
                "is_anomalous": 1,
                "anomaly_type": "dormant_withdrawer",
                "account_age_days": (ts - profile["account_created"]).days,
            }
        )
        events.append(e)

    return events


ANOMALY_GENERATORS = {
    "brute_forcer": events_brute_forcer,
    "ip_hopper": events_ip_hopper,
    "wash_trader": events_wash_trader,
    "structurer": events_structurer,
    "bot_trader": events_bot_trader,
    "dormant_withdrawer": events_dormant_withdrawer,
    "deposit_withdrawal_cycler": events_deposit_withdrawal_cycler,
}


def generate_dataset():
    print("Building user profiles....")
    profiles = build_user_profiles()
    user_ids = list(profiles.keys())
    anomalous_users = [uid for uid, p in profiles.items() if p["is_anomalous"]]
    all_events = []

    print(f"Injecting Anomaly patterns for {len(anomalous_users)} users....")
    for uid in anomalous_users:
        profile = profiles[uid]
        generator = ANOMALY_GENERATORS[profile["anomaly_type"]]
        all_events.extend(generator(uid, profile))

    # Step 2: fill remaining quota with normal events
    remaining = N_EVENTS - len(all_events)
    print(f"Generating {remaining} normal events...")
    for _ in range(remaining):
        uid = random.choice(user_ids)
        profile = profiles[uid]
        e = generate_normal_event(uid, profile)
        # Anomalous users' normal events are also labelled so the model sees full context
        if profile["is_anomalous"]:
            e["is_anomalous"] = 1
            e["anomaly_type"] = profile["anomaly_type"]
        all_events.append(e)

    # Step 3: sort chronologically and save
    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)

    Path("output_csv").mkdir(exist_ok=True)
    out_path = "output_csv/events.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"Saved to          : {out_path}")
    print(f"Total events      : {len(df)}")
    print(f"Total columns     : {len(df.columns)}")
    print(f"Anomalous users   : {len(anomalous_users)} / {N_USERS}")
    print(f"\nEvent type breakdown:")
    print(df["event_type"].value_counts().to_string())
    print(f"\nAnomaly type breakdown:")
    print(df[df["is_anomalous"] == 1]["anomaly_type"].value_counts().to_string())
    print(f"{'='*50}")
    return df


if __name__ == "__main__":
    generate_dataset()
