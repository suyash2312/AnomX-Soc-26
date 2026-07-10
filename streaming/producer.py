# streaming/producer.py
#
# Reads features.csv sorted by timestamp and publishes each event
# as a JSON message to the Redpanda topic.
#
# Run:
#   python streaming/producer.py
#
# Make sure Redpanda is running first:
#   docker compose up -d

import json
import time
import math
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer
from streaming.stream_config import (
    REDPANDA_BROKER,
    TOPIC_NAME,
    PUBLISH_DELAY_SEC,
    FEATURES_PATH,
)

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / FEATURES_PATH


def clean_event(row: dict) -> dict:
    """
    Convert a dataframe row to a JSON-safe dict.
    Replaces NaN and Inf with 0.0 so json.dumps doesn't fail.
    """
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[k] = 0.0
        else:
            cleaned[k] = v
    return cleaned


def run():
    print(f"Loading events from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Total events to publish: {len(df)}")
    print(f"  Publishing to topic    : {TOPIC_NAME}")
    print(f"  Broker                 : {REDPANDA_BROKER}")
    print(f"  Delay between events   : {PUBLISH_DELAY_SEC * 1000:.0f}ms")
    print()

    # connect to Redpanda
    producer = KafkaProducer(
        bootstrap_servers=REDPANDA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        # retry a few times if broker is briefly unavailable
        retries=5,
        retry_backoff_ms=500,
    )

    published  = 0
    anomalies  = 0

    try:
        for _, row in df.iterrows():
            event = clean_event(row.to_dict())

            # use user_id as the partition key so events from the same user
            # always go to the same partition (preserves per-user ordering)
            key = str(event.get("user_id", "unknown")).encode("utf-8")

            producer.send(TOPIC_NAME, key=key, value=event)
            published += 1

            is_anom = int(event.get("is_anomalous", 0))
            if is_anom:
                anomalies += 1

            # print progress every 100 events
            if published % 100 == 0:
                print(f"  Published {published}/{len(df)} events  |  anomalies so far: {anomalies}")

            time.sleep(PUBLISH_DELAY_SEC)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        producer.flush()
        producer.close()
        print(f"\nDone. Published {published} events ({anomalies} anomalous).")


if __name__ == "__main__":
    run()
