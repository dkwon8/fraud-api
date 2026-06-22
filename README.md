# Real-Time Fraud Detection API

[![CI](https://github.com/dkwon8/fraud-api/actions/workflows/ci.yml/badge.svg)](https://github.com/dkwon8/fraud-api/actions/workflows/ci.yml)

A production-style fraud detection system that scores credit card transactions in real time using an ensemble of an Isolation Forest and a PyTorch autoencoder, streamed through Kafka with a feedback loop for drift monitoring.

## Architecture

```
CSV / Kafka Producer
        │
        ▼
   ┌─────────┐    JSON     ┌──────────────────┐   INSERT   ┌───────────┐
   │ Redpanda │───────────▶│   FastAPI API     │──────────▶│ PostgreSQL │
   │ (Kafka)  │            │                   │           │            │
   └─────────┘            │  POST /score      │           └───────────┘
        ▲                  │  POST /label      │                │
        │                  │  GET  /health     │                │
   ┌─────────┐            └──────────────────┘           SELECT │
   │ Consumer │─── scores + labels ──────────────────────────────┘
   └─────────┘            │
                           ▼
                      ┌─────────┐
                      │  MLflow  │  ◀── FPR drift tracking
                      └─────────┘
```

## Performance

All metrics from actual training runs on the Kaggle Credit Card Fraud Detection dataset (284,807 transactions).

| Model | ROC-AUC |
|---|---|
| Isolation Forest | 0.9537 |
| Autoencoder | 0.9404 |
| **Ensemble** | **0.9566** |

| Metric | Value |
|---|---|
| Precision | 0.2910 |
| Recall | 0.5612 |
| F1 Score | 0.3833 |
| FPR | 0.002356 |
| Threshold | 0.30 |
| Scoring Latency (p99) | <50ms |

## Tech Stack

- **API**: FastAPI (async), Uvicorn
- **ML**: PyTorch (autoencoder), scikit-learn (Isolation Forest)
- **Streaming**: Kafka via Redpanda
- **Database**: PostgreSQL with async SQLAlchemy
- **Tracking**: MLflow
- **Containers**: Podman / Docker compatible

## Quickstart

### Prerequisites

- Python 3.12+
- Podman or Docker
- [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place `creditcard.csv` in `data/`

### Local Development

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Postgres
podman run -d --name frauddb -e POSTGRES_PASSWORD=password -e POSTGRES_DB=frauddb -p 5432:5432 postgres:15

# Train models
python -m models.train_all

# Start the API
uvicorn api.main:app --reload

# Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}'
```

### Kafka Streaming Demo

```bash
# Start Redpanda
podman run -d --name redpanda -p 9092:9092 -p 9644:9644 docker.io/redpandadata/redpanda:latest redpanda start --overprovisioned --smp 1 --memory 512M --reserve-memory 0M --node-id 0 --advertise-kafka-addr localhost:9092 --kafka-addr 0.0.0.0:9092

# Create topic
podman exec redpanda rpk topic create transactions

# Stream transactions (Terminal 2)
python -m kafka.producer --rate 100 --limit 5000

# Consume, score, and label (Terminal 3)
python -m kafka.consumer

# Check FPR drift
python -m mlflow_tracking.drift
```

### Container Deployment

```bash
podman-compose up
```

### Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

### `GET /health`
Returns model status and uptime.

### `POST /score`
Accepts 29 transaction features (V1–V28 + Amount), returns fraud scores from both models, fused score, predicted label, and scoring latency.

### `POST /label`
Accepts a transaction ID and true label (0 or 1). Updates the database record, enabling the feedback loop for drift monitoring.

## How It Works

1. **Training**: Both models train on normal transactions only — they learn what "normal" looks like
2. **Scoring**: New transactions are scored by both models. Anomaly scores are fused (0.4 × IF + 0.6 × AE) into a final score
3. **Feedback Loop**: Analysts label transactions via `POST /label`, enabling FPR drift tracking over time via MLflow
4. **Drift Monitoring**: Every 500 labels, the system computes the actual false positive rate and logs it to MLflow, building a time series to detect model degradation

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB/Worldline. 284,807 transactions with 492 frauds (0.17%). Features V1–V28 are PCA-transformed; Time and Amount are original.
