import uuid
import pytest

VALID_TRANSACTION = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62,
}


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


@pytest.mark.anyio
async def test_score_valid(client):
    resp = await client.post("/score", json=VALID_TRANSACTION)
    assert resp.status_code == 200
    data = resp.json()
    assert "transaction_id" in data
    assert 0 <= data["if_score"] <= 1
    assert 0 <= data["ae_score"] <= 1
    assert 0 <= data["final_score"] <= 1
    assert data["predicted_label"] in (0, 1)
    assert "latency_ms" in data


@pytest.mark.anyio
async def test_score_invalid_payload(client):
    resp = await client.post("/score", json={"bad": "data"})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_label_unknown_id(client):
    fake_id = str(uuid.uuid4())
    resp = await client.post("/label", json={
        "transaction_id": fake_id,
        "true_label": 0,
    })
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_score_then_label(client):
    # Score a transaction
    score_resp = await client.post("/score", json=VALID_TRANSACTION)
    assert score_resp.status_code == 200
    txn_id = score_resp.json()["transaction_id"]

    # Label it
    label_resp = await client.post("/label", json={
        "transaction_id": txn_id,
        "true_label": 1,
    })
    assert label_resp.status_code == 200
    assert label_resp.json()["true_label"] == 1


@pytest.mark.anyio
async def test_label_invalid_value(client):
    fake_id = str(uuid.uuid4())
    resp = await client.post("/label", json={
        "transaction_id": fake_id,
        "true_label": 5,
    })
    assert resp.status_code == 422
