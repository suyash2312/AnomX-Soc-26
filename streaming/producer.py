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

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / FEATURES_PATH


def clean_event(row: dict) -> dict:
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

    producer = KafkaProducer(
        bootstrap_servers=REDPANDA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=5,
        retry_backoff_ms=500,
    )

    published = 0
    anomalies = 0

    try:
        for _, row in df.iterrows():
            event = clean_event(row.to_dict())

            key = str(event.get("user_id", "unknown")).encode("utf-8")

            producer.send(TOPIC_NAME, key=key, value=event)
            published += 1

            is_anom = int(event.get("is_anomalous", 0))
            if is_anom:
                anomalies += 1

            if published % 100 == 0:
                print(
                    f"  Published {published}/{len(df)} events  |  anomalies so far: {anomalies}"
                )

            time.sleep(PUBLISH_DELAY_SEC)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        producer.flush()
        producer.close()
        print(f"\nDone. Published {published} events ({anomalies} anomalous).")


if __name__ == "__main__":
    run()
