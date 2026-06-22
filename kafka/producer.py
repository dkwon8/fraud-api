"""
Simulate a live transaction stream by reading the CSV and publishing to Kafka.

Usage:
    python -m kafka.producer
    python -m kafka.producer --rate 200 --limit 10000
"""
import argparse
import asyncio
import json
import os
import time

import pandas as pd
from aiokafka import AIOKafkaProducer
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "transactions"


async def produce(csv_path: str, rate: int, limit: int):
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    await producer.start()

    try:
        df = pd.read_csv(csv_path)
        count = 0
        delay = 1.0 / rate

        print(f"Publishing to '{TOPIC}' at {rate} txn/sec (limit={limit})")

        for _, row in df.iterrows():
            if count >= limit:
                break

            message = {
                "V1": row["V1"], "V2": row["V2"], "V3": row["V3"],
                "V4": row["V4"], "V5": row["V5"], "V6": row["V6"],
                "V7": row["V7"], "V8": row["V8"], "V9": row["V9"],
                "V10": row["V10"], "V11": row["V11"], "V12": row["V12"],
                "V13": row["V13"], "V14": row["V14"], "V15": row["V15"],
                "V16": row["V16"], "V17": row["V17"], "V18": row["V18"],
                "V19": row["V19"], "V20": row["V20"], "V21": row["V21"],
                "V22": row["V22"], "V23": row["V23"], "V24": row["V24"],
                "V25": row["V25"], "V26": row["V26"], "V27": row["V27"],
                "V28": row["V28"], "Amount": row["Amount"],
                "true_label": int(row["Class"]),
            }

            await producer.send_and_wait(TOPIC, message)
            count += 1

            if count % 500 == 0:
                print(f"  Published {count}/{limit}")

            await asyncio.sleep(delay)

        print(f"Done. Published {count} transactions.")

    finally:
        await producer.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/creditcard.csv")
    parser.add_argument("--rate", type=int, default=100, help="Transactions per second")
    parser.add_argument("--limit", type=int, default=5000, help="Max transactions to send")
    args = parser.parse_args()

    asyncio.run(produce(args.csv, args.rate, args.limit))


if __name__ == "__main__":
    main()
