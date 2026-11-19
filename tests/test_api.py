import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import JsonFormatter, create_app, logger


class StubModel:
    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return np.full(len(frame), 1234.5)


@pytest.fixture
def client():
    model = StubModel()
    with TestClient(create_app(model=model)) as test_client:
        yield test_client, model


def test_health_and_prediction(client, request_body):
    api, model = client
    assert api.get("/health").json() == {"status": "ok"}
    response = api.post("/predict", json=request_body)
    assert response.status_code == 200
    assert response.json() == {"prediction": 1234.5}
    assert model.calls == 1


@pytest.mark.parametrize("field,value", [
    ("rain_mm", 100.01), ("ticket_price_rs", 4.99), ("hour", 24),
    ("nearby_offices", -1), ("line", "unknown"),
    ("temp_c", "NaN"),
    ("gate_count_end_of_hour", 1234), ("entries_per_hour", 1234),
])
def test_invalid_input_never_reaches_model(client, request_body, field, value):
    api, model = client
    response = api.post("/predict", json={**request_body, field: value})
    assert response.status_code == 422
    assert model.calls == 0


def test_client_bounds_are_inclusive(client, request_body):
    api, _ = client
    response = api.post("/predict", json={**request_body, "rain_mm": 100, "ticket_price_rs": 5})
    assert response.status_code == 200


def test_missing_field_is_rejected(client, request_body):
    api, model = client
    del request_body["hour"]
    assert api.post("/predict", json=request_body).status_code == 422
    assert model.calls == 0


def test_prediction_log_is_json_without_request_fields(client, request_body):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        api, _ = client
        assert api.post("/predict", json=request_body).status_code == 200
        event = json.loads(stream.getvalue())
        assert event["event"] == "prediction"
        assert event["prediction"] == 1234.5
        assert "latency_ms" in event
        assert not set(request_body).intersection(event)
    finally:
        logger.removeHandler(handler)
