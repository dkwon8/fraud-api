"""
Consume transactions from Kafka, score via the API, then label (feedback loop).

Usage:
    python -m kafka.consumer
"""
import asyncio
import json
import os

import httpx
from aiokafka import AIOKafkaConsumer
from dotenv import load_dotenv

load_dotenv()

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
TOPIC = "transactions"


async def consume():
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="fraud-scorer",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    await consumer.start()

    scored = 0
    correct = 0

    try:
        async with httpx.AsyncClient(base_url=API_BASE, timeout=10.0) as client:
            async for msg in consumer:
                txn = msg.value
                true_label = txn.pop("true_label")

                # Score the transaction via the API
                resp = await client.post("/score", json=txn)
                if resp.status_code != 200:
                    print(f"  Score failed: {resp.status_code} {resp.text}")
                    continue

                result = resp.json()
                txn_id = result["transaction_id"]
                predicted = result["predicted_label"]
                final_score = result["final_score"]

                # Brief delay to simulate analyst review time
                await asyncio.sleep(0.01)

                # Label the transaction (feedback loop)
                label_resp = await client.post("/label", json={
                    "transaction_id": txn_id,
                    "true_label": true_label,
                })

                scored += 1
                if predicted == true_label:
                    correct += 1

                if scored % 100 == 0:
                    accuracy = correct / scored * 100
                    print(
                        f"  Scored {scored} | "
                        f"Latest: score={final_score:.4f} pred={predicted} actual={true_label} | "
                        f"Accuracy: {accuracy:.1f}%"
                    )

    finally:
        await consumer.stop()
        if scored > 0:
            print(f"\nFinal: {scored} scored, {correct/scored*100:.1f}% accuracy")


def main():
    asyncio.run(consume())


if __name__ == "__main__":
    main()
