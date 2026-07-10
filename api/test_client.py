import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    print("── GET /health ────────────────────────────────────────")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print()


def test_normal_event():
    """A regular deposit from a normal user — should come back as normal."""
    print("── POST /score  (normal deposit) ──────────────────────")
    event = {
        "user_id":    "USER_0001",
        "event_type": "deposit",
        "amount":     500.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend":  0,
        "account_age_days": 120,
        # most features default to 0.0 — normal user has no suspicious signals
    }
    r = requests.post(f"{BASE_URL}/score", json=event)
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print()


def test_wash_trader():
    """A wash trader event — very high volume, always profitable."""
    print("── POST /score  (wash trader) ─────────────────────────")
    event = {
        "user_id":                   "USER_0007",
        "event_type":                "trade",
        "trade_volume":              180000.0,
        "trade_volume_vs_baseline":  15.3,
        "pnl":                       4200.0,
        "margin_used":               95000.0,
        "lot_size":                  35.0,
        "trade_duration_seconds":    12.0,
        "roll_5_trade_vol_mean":     160000.0,
        "roll_5_pnl_mean":           3800.0,
        "trade_vol_zscore":          8.4,
        "pnl_zscore":                6.1,
        "trade_vol_vs_baseline_zscore": 7.2,
        "hour_of_day":               3,
        "is_night_trade":            1,
        "account_age_days":          200,
    }
    r = requests.post(f"{BASE_URL}/score", json=event)
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print()


def test_brute_forcer():
    """Brute force login attempt — many failed logins building up."""
    print("── POST /score  (brute forcer) ────────────────────────")
    event = {
        "user_id":                  "USER_0003",
        "event_type":               "login",
        "failed_attempts":          7.0,
        "rolling_failed_attempts_5": 28.0,
        "burst_count_5min":         8,
        "burst_count_30min":        9,
        "login_success":            1.0,
        "unique_ips_last_10_logins": 6.0,
        "timezone_gap_hours":       9.5,
        "rolling_timezone_gap_5":   8.2,
        "hour_of_day":              2,
        "account_age_days":         90,
    }
    r = requests.post(f"{BASE_URL}/score", json=event)
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print()


def test_deposit_withdrawal_cycler():
    """Deposit followed immediately by withdrawal — money laundering pattern."""
    print("── POST /score  (deposit-withdrawal cycler) ───────────")
    event = {
        "user_id":                       "USER_0015",
        "event_type":                    "withdrawal",
        "amount":                        9500.0,
        "withdrawal_to_deposit_ratio":   0.95,
        "is_immediate_withdrawal":       1.0,
        "rolling_immediate_withdrawal_5": 4.0,
        "roll_5_deposit_sum":            10000.0,
        "time_since_last_deposit_sec":   3600.0,
        "amount_zscore":                 5.2,
        "is_large_withdrawal":           1,
        "account_age_days":              60,
        "hour_of_day":                   11,
    }
    r = requests.post(f"{BASE_URL}/score", json=event)
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print()


def test_bot_trader():
    """Bot session — inhuman click rate at 3 AM."""
    print("── POST /score  (bot trader) ──────────────────────────")
    event = {
        "user_id":                "USER_0021",
        "event_type":             "session",
        "click_rate_per_min":     650.0,
        "session_duration_mins":  0.4,
        "page_clicks":            260.0,
        "roll_5_click_rate_mean": 580.0,
        "click_rate_zscore":      12.3,
        "session_duration_zscore": -2.1,
        "burst_count_30min":      8,
        "hour_of_day":            3,
        "account_age_days":       45,
    }
    r = requests.post(f"{BASE_URL}/score", json=event)
    print(f"Status  : {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  ForexGuard API Test Client")
    print("=" * 55)
    print()

    try:
        test_health()
        test_normal_event()
        test_wash_trader()
        test_brute_forcer()
        test_deposit_withdrawal_cycler()
        test_bot_trader()
        print("=" * 55)
        print("  All tests done.")
        print("=" * 55)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server.")
        print("Make sure the API is running:")
        print("  uvicorn api.main:app --reload")
