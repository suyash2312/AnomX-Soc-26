# streaming/stream_config.py
#
# Shared config for producer and consumer.
# Change REDPANDA_BROKER if running on a different host/port.

REDPANDA_BROKER = "localhost:19092"   # mapped port from docker-compose.yml
TOPIC_NAME      = "anomx-events"
GROUP_ID        = "anomx-consumer-group"

# how long producer waits between publishing events (seconds)
# 0.05 = 50ms, set to 0 to replay as fast as possible
PUBLISH_DELAY_SEC = 0.05

# path to features.csv relative to project root
FEATURES_PATH = "Week - 4/data/features.csv"
