import os
import time
from contextlib import asynccontextmanager
from uuid import UUID

import joblib
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

from db.connection import engine, get_db
from db.models import Base, Transaction
from api.schemas import (
    ScoreRequest, ScoreResponse, LabelRequest, LabelResponse, HealthResponse,
)
from models.preprocess import preprocess_single, FEATURE_COLS
from models.autoencoder import FraudAutoencoder, reconstruction_error
from models.scorer import fuse_scores, THRESHOLD

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")

# Module-level dict to hold loaded models — shared across all requests
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup (load models) and once at shutdown (close DB)."""

    # Create the transactions table if it doesn't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Load all model artifacts into memory
    _state["scaler"] = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    _state["if_model"] = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    _state["bounds"] = joblib.load(os.path.join(ARTIFACTS_DIR, "norm_bounds.joblib"))

    checkpoint = torch.load(
        os.path.join(ARTIFACTS_DIR, "autoencoder.pt"),
        map_location="cpu", weights_only=True,
    )
    ae = FraudAutoencoder(input_dim=checkpoint["input_dim"])
    ae.load_state_dict(checkpoint["model_state_dict"])
    ae.eval()
    _state["ae_model"] = ae
    _state["device"] = torch.device("cpu")
    _state["start_time"] = time.time()

    yield  # App runs here, handling requests

    await engine.dispose()


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded="if_model" in _state and "ae_model" in _state,
        uptime_seconds=round(time.time() - _state.get("start_time", time.time()), 2),
    )


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest, db: AsyncSession = Depends(get_db)):
    t0 = time.perf_counter()

    # Preprocess: build feature array, scale Amount
    features = req.model_dump()
    X = preprocess_single(features, _state["scaler"])

    # Isolation Forest score: raw → flip sign → normalize using training bounds
    if_raw = -_state["if_model"].decision_function(X)
    bounds = _state["bounds"]
    if_score = float(np.clip(
        (if_raw[0] - bounds["if_min"]) / (bounds["if_max"] - bounds["if_min"] + 1e-10),
        0, 1,
    ))

    # Autoencoder score: reconstruction error → normalize using training bounds
    ae_error = reconstruction_error(_state["ae_model"], X, _state["device"])
    ae_score = float(np.clip(
        (ae_error[0] - bounds["ae_min"]) / (bounds["ae_max"] - bounds["ae_min"] + 1e-10),
        0, 1,
    ))

    # Fuse scores and get prediction
    result = fuse_scores(if_score, ae_score)

    # Store in database
    txn = Transaction(
        features=features,
        if_score=result.if_score,
        ae_score=result.ae_score,
        final_score=result.final_score,
        predicted_label=result.predicted_label,
    )
    db.add(txn)
    await db.commit()
    await db.refresh(txn)

    latency_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        transaction_id=txn.id,
        if_score=result.if_score,
        ae_score=result.ae_score,
        final_score=result.final_score,
        predicted_label=result.predicted_label,
        threshold_used=THRESHOLD,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/label", response_model=LabelResponse)
async def label(req: LabelRequest, db: AsyncSession = Depends(get_db)):
    # Find the transaction by ID (str() ensures compatibility with both Postgres and SQLite)
    result = await db.execute(
        select(Transaction).where(Transaction.id == str(req.transaction_id))
    )
    txn = result.scalar_one_or_none()

    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # Update the true label (the feedback loop)
    txn.true_label = req.true_label
    await db.commit()

    return LabelResponse(
        transaction_id=txn.id,
        true_label=req.true_label,
        message="Label recorded",
    )
