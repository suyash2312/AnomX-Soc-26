# AnomX - Midterm Submission
**Seasons of Code 2026 | Suyash Jagtap**

---

## Repository Structure

```
AnomX-SoC-26/
├── README.md
├── Week - 3/
│   ├── generate_events.py
│   └── data/
│       └── events.csv
└── Week - 4/
    ├── feature_engineering.py
    └── data/
        └── features.csv
```

---

## Week 1-2: Python, Git & Foundations

### Python Basics
Covered the basics needed for this project — data types, loops, functions, and how to work with libraries like pandas and numpy. Most of this was already familiar but going through it properly helped understand how the data pipeline is structured.

### Git & GitHub
Learned how to use git for version control. The basic workflow I followed throughout this project:
```bash
git add .
git commit -m "message"
git push origin main
```
Key thing I understood — commits should be small and meaningful, not one big dump at the end.

### Pandas and NumPy
These are used everywhere in this project:
- **Pandas** — loading CSVs, filtering by event type, groupby, rolling windows
- **NumPy** — generating random values with `np.random.normal()`, `np.where()` for conditional columns

### What is Anomaly Detection
Anomaly detection means finding data points that don't follow the normal pattern. In this project, we're looking at financial platform activity and trying to flag users who are doing something suspicious — like logging in from 6 different countries in 10 minutes, or placing trades that are 15x their normal size.

The important thing I learned here is that a single event is usually not enough to call something anomalous. It's the pattern over time that matters. That's why feature engineering is needed — raw events don't carry enough context.

---

## Week 3: Event Generation

### What the script does
`generate_events.py` generates a synthetic dataset of financial platform activity for 200 users over 3 months (Jan–Mar 2024). 30 out of 200 users (15%) are anomalous.

The flow is:
1. Build a profile for each user (home country, typical trade size, typical deposit, preferred device, etc.)
2. For anomalous users, inject a specific fraud pattern
3. Fill the rest with normal events generated from each user's profile

The reason for building user profiles first is that anomalies are defined relative to the user's own behaviour, not globally. A $30,000 trade is normal for one user but suspicious for another.

### Event Types

| Event Type | What it captures |
|---|---|
| `login` | IP, country, device, success/failure, timezone gap from home |
| `trade` | Instrument, volume, PnL, margin used, lot size, duration |
| `deposit` | Amount, method, immediate withdrawal flag |
| `withdrawal` | Amount, method, immediate withdrawal flag |
| `session` | Duration, page clicks, click rate per minute |
| `kyc_change` | Type of KYC update |

### Anomaly Types

| Anomaly | What it does | Key signal |
|---|---|---|
| `brute_forcer` | 4-8 failed logins every 30 seconds then success | `failed_attempts` keeps going up |
| `ip_hopper` | Logs in from 6-10 different countries within minutes | `country` and `timezone_gap_hours` |
| `wash_trader` | Trades 10-20x normal volume, always profitable | `trade_volume_vs_baseline`, `pnl` always positive |
| `structurer` | 10-15 deposits all between $490-$999 | `amount` stuck just under $1000 |
| `bot_trader` | 300-800 clicks/min at 2-4 AM | `click_rate_per_min`, `hour_of_day` |
| `dormant_withdrawer` | Long inactivity then sudden huge withdrawal | `amount` very high, old account |
| `deposit_withdrawal_cycler` | Deposits then withdraws 85-98% of it within hours | `is_immediate_withdrawal` |

### Dataset
- 5000 events, 200 users, 31 columns
- Date range: Jan 2024 - Mar 2024
- ~19% events are anomalous

Some events which are suspicious are mentioned here and considered as anomaly:
- `timezone_gap_hours` — difference between login country timezone and user's home timezone. 0 for normal logins, 8-13 for ip_hopper
- `trade_volume_vs_baseline` — actual volume / user's typical volume. Near 1.0 for normal, 10-20 for wash_trader
- `is_immediate_withdrawal` — 1 if withdrawal happens within hours of a deposit
- `is_night_trade` — 1 if trade placed between midnight and 5 AM local time

---

## Week 4: Feature Engineering & EDA

### Why features are needed
Raw events tell you what happened. Features tell you whether it's normal or not. For example, a login from Russia is just a login. But 6 logins from 6 different countries in 10 minutes is clearly suspicious. Features capture this kind of context.

`feature_engineering.py` takes `events.csv` (31 columns) and adds 36 new columns → `features.csv` (67 columns total).

### Features built

**Time-delta features** — how fast are events happening?
- `time_since_last_event_sec` — seconds since previous event. Near zero = too fast.
- `time_since_last_login_sec` — useful for brute_forcer (logins every 30s)
- `time_since_last_deposit_sec` — useful for cycler (withdrawal comes right after deposit)

**Rolling trade features** — is trading behaviour unusual recently?
- `roll_5_trade_vol_mean` — avg volume over last 5 trades. Wash trader: 27,585 vs normal: 3,524
- `roll_5_pnl_mean` — avg PnL over last 5 trades. Wash trader is always positive.
- `roll_5_margin_mean` — margin used. Wash traders use 50k-200k vs normal 100-5000.
- (same features for window=10)

**Rolling session features** — is click behaviour robotic?
- `roll_5_click_rate_mean` — bot_trader avg: 178/min vs normal: 0.30/min
- `roll_5_session_dur_mean` — bot_trader sessions are very short (0.2-1.5 min)

**Burst count features** — too many events in a short window?
- `burst_count_5min` — events in last 5 min. Brute_forcer spikes this.
- `burst_count_30min` — events in last 30 min. Bot_trader and ip_hopper show up here.

**Login anomaly features** — multiple locations or failed attempts?
- `unique_countries_last_10_logins` — ip_hopper reaches 6-8, normal users stay at 1-2
- `unique_ips_last_10_logins` — new IP every login for ip_hopper
- `rolling_failed_attempts_5` — brute_forcer avg: 4.04 vs normal: 0.03
- `rolling_timezone_gap_5` — avg timezone gap across last 5 logins. High = impossible travel.

**Financial ratio features** — withdrawing right after depositing?
- `roll_5_deposit_sum` — total deposited in last 5 deposits
- `withdrawal_to_deposit_ratio` — cycler deposits 10k, withdraws 9.5k → ratio ~0.95 (capped at 10)
- `rolling_immediate_withdrawal_5` — cycler avg: 1.24 vs normal: 0.00

**Structurer features** — many similar small deposits?
- `deposit_amount_rolling_std` — structurer has very low std (all deposits 490-999)
- `deposit_count_last_10` — structurer makes 10-15 deposits in a burst
- `deposit_amount_rolling_mean` — stays around 740 for structurer

**Dormant withdrawer features**
- `is_large_withdrawal` — 1 if withdrawal > $5000
- `withdrawal_amount_vs_account_age` — large amount on an old dormant account

**Z-score features** — is this value unusual for this specific user?
Per-user z-scores are more useful than global ones because user behaviour varies a lot.
- `trade_vol_zscore` — wash_trader's volume is 10-20x their own avg → z > 5
- `click_rate_zscore` — bot_trader hits z > 10
- `amount_zscore` — dormant_withdrawer: one huge withdrawal with no history → extreme z
- `pnl_zscore`, `session_duration_zscore`, `trade_vol_vs_baseline_zscore`

### Observations from EDA

| Feature | Normal users | Anomalous |
|---|---|---|
| `rolling_failed_attempts_5` | 0.03 | 4.04 (brute_forcer) |
| `roll_5_trade_vol_mean` | 3,524 | 27,585 (wash_trader) |
| `roll_5_pnl_mean` | -0.34 | 709 (wash_trader) |
| `roll_5_click_rate_mean` | 0.30 | 178 (bot_trader) |
| `rolling_immediate_withdrawal_5` | 0.00 | 1.24 (cycler) |

The signals are clearly separable which is a good sign.

One thing I noticed — `withdrawal_to_deposit_ratio` in the reference implementation can produce extremely large values (up to 2 trillion) due to division by near-zero. I capped it at 10 which is cleaner for model training.

### Key things I learned
- Single event values don't mean much. Rolling windows and burst counts give context.
- Per-user z-scores work better than global ones for catching anomalies relative to a user's own baseline.
- Some anomalies need multiple features together — structurer is best caught by combining low std + high deposit count + mean amount near 740. No single feature is enough.
- NaN filling with 0 makes sense here — 0 means "no signal yet" for users with no events of that type.

---

*Mentor: Sugandh Kumar | SoC 2026*

