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

**How it works:**
1. **Training (offline):** Both models train on normal transactions only — they learn what "normal" looks like. Anything that deviates gets a high anomaly score.
2. **Scoring (real-time):** New transactions are scored by both models. Scores are fused (0.4 × Isolation Forest + 0.6 × Autoencoder) into a final fraud score between 0 and 1.
3. **Feedback loop:** After scoring, analysts can label transactions as fraud or normal via `POST /label`. This updates the database record.
4. **Drift monitoring:** Every 500 labels, the system computes the actual false positive rate and logs it to MLflow. A rising FPR over time signals the model needs retraining.

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
| Scoring Latency | ~23ms |

## Tech Stack

- **API:** FastAPI (async), Uvicorn
- **ML:** PyTorch (autoencoder), scikit-learn (Isolation Forest)
- **Streaming:** Kafka via Redpanda
- **Database:** PostgreSQL with async SQLAlchemy
- **Tracking:** MLflow
- **Containers:** Podman / Docker compatible
- **Testing:** pytest with async support

## Project Structure

```
fraud-api/
├── api/                  # FastAPI application
│   ├── main.py           # App setup, lifespan, route handlers
│   └── schemas.py        # Pydantic request/response models
├── db/                   # Database layer
│   ├── models.py         # SQLAlchemy Transaction model
│   └── connection.py     # Async engine and session factory
├── models/               # ML pipeline
│   ├── preprocess.py     # Feature engineering (drop Time, scale Amount, split)
│   ├── train_if.py       # Isolation Forest training and scoring
│   ├── autoencoder.py    # PyTorch autoencoder training and scoring
│   ├── scorer.py         # Score fusion (0.4 IF + 0.6 AE) and evaluation
│   ├── train_all.py      # Orchestrator — runs the full training pipeline
│   └── artifacts/        # Saved models (scaler, IF, AE, normalization bounds)
├── kafka/                # Streaming pipeline
│   ├── producer.py       # Reads CSV, publishes transactions to Kafka
│   └── consumer.py       # Consumes from Kafka, scores via API, labels
├── mlflow_tracking/      # Monitoring
│   └── drift.py          # Computes FPR from labeled data, logs to MLflow
├── tests/                # pytest suite
│   ├── conftest.py       # Test fixtures (in-memory DB, model loading)
│   └── test_api.py       # API endpoint tests
├── notebooks/
│   └── eda.ipynb         # Exploratory data analysis
├── compose.yml           # One-command deployment (Podman/Docker)
├── Dockerfile
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.12+
- Podman or Docker
- [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Installation

```bash
git clone https://github.com/dkwon8/fraud-api.git
cd fraud-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download the Dataset

Download `creditcard.csv` from the Kaggle link above and place it in the `data/` directory:

```
fraud-api/data/creditcard.csv
```

This file is ~150MB and contains 284,807 transactions. It is gitignored.

### Start PostgreSQL

The API stores every scored transaction in Postgres. Start it in a container:

```bash
podman run -d --name frauddb \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=frauddb \
  -p 5432:5432 \
  postgres:15
```

If you use Docker, replace `podman` with `docker` — all commands are the same.

### Train the Models (~2 minutes)

This runs the full training pipeline and saves model artifacts:

```bash
python -m models.train_all
```

You'll see output for each step: preprocessing, Isolation Forest training, autoencoder training (50 epochs), and ensemble evaluation with final metrics.

The trained models are saved to `models/artifacts/`. These are already committed to the repo, so you can skip this step if you just want to run the API.

## Usage

### Start the API

```bash
uvicorn api.main:app --reload
```

The server starts on `http://localhost:8000`. On startup it loads the trained models into memory and creates the database table if it doesn't exist.

Interactive API docs are available at `http://localhost:8000/docs` (auto-generated by FastAPI).

### API Endpoints

#### `GET /health` — Check server status

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "model_loaded": true, "uptime_seconds": 16.78}
```

#### `POST /score` — Score a transaction

Send 29 features (V1–V28 + Amount) and get back fraud scores:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}'
```

Response:
```json
{
  "transaction_id": "b7ff2675-397b-4274-88b6-caf02a5e1af7",
  "if_score": 0.0613,
  "ae_score": 0.0013,
  "final_score": 0.0253,
  "predicted_label": 0,
  "threshold_used": 0.3,
  "latency_ms": 23.17
}
```

- `if_score` / `ae_score`: Individual model scores (0 = normal, 1 = anomalous)
- `final_score`: Weighted fusion (0.4 × IF + 0.6 × AE)
- `predicted_label`: 0 (normal) or 1 (fraud), based on threshold
- `latency_ms`: Time to preprocess + score + store in DB

#### `POST /label` — Label a transaction (feedback loop)

After scoring, submit the true label to enable drift monitoring:

```bash
curl -X POST http://localhost:8000/label \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "b7ff2675-397b-4274-88b6-caf02a5e1af7", "true_label": 0}'
```

Response:
```json
{"transaction_id": "b7ff2675-...", "true_label": 0, "message": "Label recorded"}
```

### Kafka Streaming Demo

This simulates a live transaction stream. You need 3 terminals running simultaneously.

**First, start Redpanda (Kafka-compatible broker):**

```bash
podman run -d --name redpanda \
  -p 9092:9092 -p 9644:9644 \
  docker.io/redpandadata/redpanda:latest \
  redpanda start --overprovisioned --smp 1 --memory 512M \
  --reserve-memory 0M --node-id 0 \
  --advertise-kafka-addr localhost:9092 \
  --kafka-addr 0.0.0.0:9092

podman exec redpanda rpk topic create transactions
```

**Terminal 1 — API server** (if not already running):
```bash
uvicorn api.main:app --reload
```

**Terminal 2 — Producer** (reads CSV, publishes to Kafka at 100 txn/sec):
```bash
python -m kafka.producer --rate 100 --limit 5000
```

**Terminal 3 — Consumer** (reads from Kafka, scores each transaction, then labels it):
```bash
python -m kafka.consumer
```

The consumer logs accuracy every 100 transactions:
```
Scored 100 | Latest: score=0.0034 pred=0 actual=0 | Accuracy: 100.0%
Scored 200 | Latest: score=0.0198 pred=0 actual=0 | Accuracy: 100.0%
...
Scored 5000 | Latest: score=0.0795 pred=0 actual=0 | Accuracy: 99.8%
```

Producer options:
- `--rate 500` — faster streaming (500 txn/sec)
- `--limit 50000` — process more of the dataset
- `--csv path/to/file.csv` — use a different data file

**After streaming, check FPR drift:**
```bash
python -m mlflow_tracking.drift
```

### Query the Database

You can inspect scored transactions directly:

```bash
# Connect to Postgres
podman exec -it frauddb psql -U postgres -d frauddb

# Inside psql:
SELECT COUNT(*) FROM transactions;
SELECT * FROM transactions WHERE predicted_label = 1;
SELECT * FROM transactions WHERE true_label = 1;
\q
```

### Container Deployment

Run the entire stack with one command:

```bash
podman-compose up
```

This starts Postgres, Redpanda, and the API together with healthchecks and dependency ordering.

### Run Tests

```bash
pytest tests/ -v
```

Tests use an in-memory SQLite database, so they don't need Postgres running.

### Stopping Services

```bash
# Stop the API: Ctrl+C in the terminal running uvicorn

# Stop and remove containers
podman stop frauddb redpanda
podman rm frauddb redpanda
```

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB/Worldline. 284,807 transactions with 492 frauds (0.17%). Features V1–V28 are PCA-transformed for confidentiality; Time and Amount are the original features.
