import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import JsonFormatter, create_app, logger


class StubModel:
    def __init__(self, value=1234.5):
        self.calls = 0
        self.value = value

    def predict(self, frame):
        self.calls += 1
        return np.full(len(frame), self.value)


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


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_outputs_are_rejected(request_body, value):
    model = StubModel(value)
    with TestClient(create_app(model=model)) as api:
        response = api.post("/predict", json=request_body)
    assert response.status_code == 500
    assert response.json() == {"detail": "Model returned a nonfinite prediction"}
    assert model.calls == 1


@pytest.mark.parametrize("value,expected", [(-12.5, 0.0), (0.0, 0.0), (1234.5, 1234.5)])
def test_finite_predictions_are_clipped_without_extra_model_calls(request_body, value, expected):
    model = StubModel(value)
    with TestClient(create_app(model=model)) as api:
        response = api.post("/predict", json=request_body)
    assert response.status_code == 200
    assert response.json() == {"prediction": expected}
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
