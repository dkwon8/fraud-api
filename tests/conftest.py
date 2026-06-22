import os
import time
import asyncio
import pytest
import joblib
import torch
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import StaticPool

from db.models import Base
from db.connection import get_db
from api.main import app, _state
from models.autoencoder import FraudAutoencoder

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def load_models():
    """Load model artifacts once for the entire test session."""
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


@pytest.fixture()
async def client():
    """Test client with a shared in-memory SQLite database."""
    # StaticPool keeps a single connection alive so data persists across sessions
    engine = create_async_engine(
        "sqlite+aiosqlite:///",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    app.dependency_overrides.clear()
    await engine.dispose()
