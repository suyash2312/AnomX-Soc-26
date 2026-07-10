import json
import sys
from pathlib import Path
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

sys.path.append(str(Path(__file__).parent.parent))

from models.scorer import ForexGuardScorer
from streaming.stream_config import (
    REDPANDA_BROKER,
    TOPIC_NAME,
    GROUP_ID,
)

SEVERITY_COLOURS = {
    "CRITICAL": "\033[91m",  # red
    "HIGH": "\033[93m",  # yellow
    "MEDIUM": "\033[94m",  # blue
    "LOW": "\033[92m",  # green
    "NONE": "\033[0m",  # reset
}
RESET = "\033[0m"


def format_alert(result: dict) -> str:
    """Format a scored event as a human-readable alert."""
    colour = SEVERITY_COLOURS.get(result["severity"], "")
    lines = []

    lines.append(f"{colour}{'='*55}{RESET}")
    lines.append(f"{colour}  {result['verdict']}  —  {result['severity']}{RESET}")
    lines.append(f"  User       : {result['user_id']}")
    lines.append(f"  Event      : {result['event_type']}")
    lines.append(f"  Score      : {result['anomaly_score']:.5f}")

    if result["reasons"]:
        lines.append(f"  Why        :")
        for r in result["reasons"]:
            lines.append(f"    • {r}")

    if result["top_features"]:
        lines.append(f"  Top features:")
        for f in result["top_features"][:3]:
            lines.append(
                f"    • {f['feature']} = {f['raw_value']} (scaled: {f['scaled_value']})"
            )

    lines.append(f"{colour}{'='*55}{RESET}")
    return "\n".join(lines)


def run():
    print("Loading ForexGuard model...")
    scorer = ForexGuardScorer()
    scorer.load()

    if not scorer.loaded:
        print("ERROR: Model not loaded. Run  python models/isolation_forest.py  first.")
        sys.exit(1)

    print(f"Connecting to Redpanda at {REDPANDA_BROKER}...")
    print(f"Subscribing to topic: {TOPIC_NAME}")
    print("Waiting for events... (Ctrl+C to stop)\n")

    try:
        consumer = KafkaConsumer(
            TOPIC_NAME,
            bootstrap_servers=REDPANDA_BROKER,
            group_id=GROUP_ID,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
    except NoBrokersAvailable:
        print("ERROR: Could not connect to Redpanda.")
        print("Make sure it's running:  docker compose up -d")
        sys.exit(1)

    processed = 0
    flagged = 0

    try:
        for message in consumer:
            event = message.value
            processed += 1

            try:
                result = scorer.score(event)
            except Exception as e:
                print(f"Scoring error on event {event.get('event_id', '?')}: {e}")
                continue

            if result["is_anomaly"]:
                flagged += 1
                print(format_alert(result))
            else:
                # print a simple dot for normal events so you can see it's running
                print(
                    f"  ✓ {event.get('user_id','?'):12s} {event.get('event_type','?'):12s} score={result['anomaly_score']:.5f}",
                    end="\r",
                )

    except KeyboardInterrupt:
        print(
            f"\n\nStopped. Processed {processed} events, flagged {flagged} anomalies."
        )
    finally:
        consumer.close()


if __name__ == "__main__":
    run()
