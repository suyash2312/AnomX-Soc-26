# AnomX - Endterm Submission
**Seasons of Code 2026 | Suyash Jagtap (24b1251)**

---
Video Link : https://drive.google.com/file/d/1v3OgduzZ0G1e0YkNZr6CwSUBE-CzJpN9/view?usp=sharing

## Repository Structure

```
AnomX-SoC-26/
├── Week - 1,2
    ├──prac.py
├── Week - 3/
│   ├── generate_events.py
│   └── data/
│       └── events.csv
├── Week - 4/
|    ├── feature_engineering.py
|    └── data/
|        └── features.csv
├── Week - 5,6/
|    ├── week-5,6_report.pdf
├── api/
|    ├── __init__.py
|    └── main.py
|    └── schemas.py
|    └── test_client.py
├── models/
|    ├── saved/
|    └── isolation_forest.py
|    └── lstm_encoder.py
|    └── scorer.py
├── streaming/
|   ├── consumer.py
|   └── producer.py
|   └── stream_config.py
├── README.md
```

---

## Week 1-2: Python, Git & Foundations

### Python Basics
Covered the basics needed for this project : data types, loops, functions, and how to work with libraries like pandas and numpy. Most of this was already familiar but going through it properly helped understand how the data pipeline is structured.

### Git & GitHub
Learned how to use git for version control. The basic workflow I followed throughout this project:
```bash
git add .
git commit -m "message"
git push origin main
```
Key thing I understood , commits should be small and meaningful, not one big dump at the end.

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

## Week 5-6: Isolation Forest

### How it works
Isolation Forest is an unsupervised anomaly detection algorithm. The idea is simple — anomalous points are rare and different, so they are easy to isolate from the rest of the data. The algorithm builds random decision trees and keeps splitting the data until each point is alone. The number of splits needed to isolate a point is called its path length.

Short path length = isolated quickly = likely anomalous. Long path length = hard to isolate = likely normal.

This is averaged across all trees to get a final anomaly score.

### Why unsupervised?
In real fraud detection you usually don't have labels for every event. Isolation Forest doesn't need labels — it just learns what normal looks like and flags anything that doesn't fit. This is more realistic than a supervised model that needs labelled data to train on.

### Implementation — `models/isolation_forest.py`
The script loads `features.csv`, drops identifier columns like `user_id` and `timestamp`, scales everything with StandardScaler, and trains the model with these settings:

- `n_estimators = 200` — more trees = more stable scores
- `contamination = 0.15` — tells the model roughly 15% of events are anomalous
- `random_state = 42` — for reproducibility

### Threshold
Instead of using sklearn's default contamination-based threshold, I set it at the **99th percentile of normal event scores**. This means only the most extreme 1% of normal-looking events get flagged. Reduces false positives significantly.

### Results

| Metric | Value |
|---|---|
| ROC-AUC | ~0.65 |
| Best detected | wash_trader, brute_forcer |
| Hardest to detect | structurer (subtle pattern) |

ROC-AUC of 0.65 is expected on a 5000-event dataset. The mentor's reference used 50,000 events — more data gives better separation. The score distribution plot shows clear separation between normal and anomalous events especially for wash_trader and brute_forcer.

### Saved artifacts
Running `isolation_forest.py` saves three files to `models/saved/`:
- `isolation_forest.pkl` — trained model
- `if_scaler.pkl` — fitted StandardScaler
- `if_threshold.pkl` — threshold value (p99 of normal scores)

---

## Week 7-8: LSTM Autoencoder

### Why LSTM after Isolation Forest?
Isolation Forest treats each event independently. It doesn't know what came before. So it can see that trade volume is high, but it can't see that this user had 7 failed logins, then logged in from a new country, then immediately placed a huge trade — that whole sequence tells a story that IF misses.

LSTM looks at sequences of events, so it understands the pattern over time.

### How the Autoencoder works
The model is trained only on normal user sequences. It learns to compress a sequence of 10 events into a small vector (encoder) and then reconstruct the original sequence back from that vector (decoder).

When a normal sequence comes in — the model reconstructs it well because it has seen this kind of pattern before. When an anomalous sequence comes in — the model can't reconstruct it well because it never learned that pattern. The reconstruction error (MSE) is high. High MSE = anomaly.

### Architecture — `models/lstm_autoencoder.py`

```
Input sequence (10 events × 55 features)
        ↓
   LSTM Encoder  →  Latent vector (32 dimensions)
        ↓
   LSTM Decoder  →  Reconstructed sequence
        ↓
   MSE between input and reconstruction = anomaly score
```

Settings used:
- Sequence length: 10 events per user
- Hidden size: 64
- Latent size: 32
- Layers: 2
- Epochs: 30
- Trained on normal sequences only

### Threshold
Set at the 95th percentile of reconstruction errors on normal training sequences. Anything above this gets flagged as anomalous.

### Saved artifacts
- `lstm_autoencoder.pt` — model weights
- `lstm_scaler.pkl` — fitted scaler
- `lstm_threshold.pkl` — MSE threshold
- `lstm_config.pkl` — model config (input size, hidden size etc.)

---

## Week 9: Streaming with Redpanda

### What is streaming and why it matters
In a real brokerage, events don't come as a CSV file — they come one by one in real time. A user logs in, that's one event. They place a trade, that's another. The system needs to score each event as it arrives, not after collecting everything.

That's what the streaming pipeline does.

### Setup
Redpanda is a Kafka-compatible message broker. It runs locally via Docker:

```bash
docker compose up -d
```

This starts a single-node Redpanda broker and automatically creates the `anomx-events` topic.

### Producer — `streaming/producer.py`
Reads `features.csv` sorted by timestamp and publishes each event as a JSON message to the `anomx-events` topic. There's a 50ms delay between events to simulate real time. Uses `user_id` as the partition key so events from the same user always go to the same partition — this preserves per-user ordering.

```bash
python streaming/producer.py
```

### Consumer — `streaming/consumer.py`
Subscribes to the `anomx-events` topic, passes each event to `ForexGuardScorer`, and prints a formatted alert whenever something is flagged. Normal events just show a dot so you can see it's running.

```bash
python streaming/consumer.py
```

### Config — `streaming/stream_config.py`
Shared settings like broker address, topic name, and publish delay. Both producer and consumer import from here so there's no duplication.

---

## Week 10: FastAPI

### What it does
FastAPI turns the trained model into an API that any client can call. Instead of running a Python script manually, you send an HTTP request and get back a JSON response with the anomaly verdict.

### Files

**`models/scorer.py` — ForexGuardScorer**
This is the core class that handles everything. It loads the trained model at startup, takes an event as a dict, builds the feature vector, scales it, runs it through the model, and returns the result. Used by both the API and the consumer.

**`api/schemas.py` — Pydantic models**
Defines what a valid request looks like. All 55 features are listed with their types. FastAPI uses this to automatically validate incoming requests — if you send wrong data types it returns a 422 error before the model even runs.

**`api/main.py` — FastAPI app**
Two endpoints:

- `GET /health` — checks if server is up and model is loaded
- `POST /score` — takes an event, returns anomaly score and verdict

Model is loaded once at startup, not on every request.

### Running it

```bash
uvicorn api.main:app --reload
```

Docs available at `http://localhost:8000/docs` — you can test the API directly from the browser without writing any code.

### Example response

```json
{
  "user_id": "USER_0007",
  "event_type": "trade",
  "anomaly_score": 0.08432,
  "is_anomaly": true,
  "severity": "CRITICAL",
  "verdict": "🚨 ANOMALY",
  "reasons": [
    "Trade volume far above user's normal",
    "Unusually large trade volume",
    "Abnormal profit/loss pattern"
  ],
  "top_features": [
    {"feature": "trade_vol_zscore", "raw_value": 8.4, "scaled_value": 6.21},
    {"feature": "roll_5_trade_vol_mean", "raw_value": 160000.0, "scaled_value": 5.87}
  ]
}
```

### Severity levels

| Severity | Score threshold |
|---|---|
| CRITICAL | > 0.06 |
| HIGH | > 0.03 |
| MEDIUM | > 0.01 |
| LOW | ≤ 0.01 |

Thresholds are based on the percentile distribution of training anomaly scores — not arbitrary numbers.

### Testing — `api/test_client.py`
Runs 5 test cases covering a normal deposit and all major anomaly types. Shows the contrast between a normal event returning `✅ NORMAL` and a wash trader returning `🚨 ANOMALY CRITICAL`.

```bash
python api/test_client.py
```

---

## How to run everything

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. generate data
python "Week - 3/generate_events.py"

# 3. run feature engineering
python "Week - 4/feature_engineering.py"

# 4. train model
python models/isolation_forest.py

# 5. start API
uvicorn api.main:app --reload

# 6. test API
python api/test_client.py

# 7. streaming (needs Docker)
docker compose up -d
python streaming/producer.py   # terminal 1
python streaming/consumer.py   # terminal 2
```

---

* Mentor: Sugandh Kumar | SoC 2026*
