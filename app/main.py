"""Serve the exported champion with the client's request validation rules."""

import json
import logging
import math
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mp135.config import ROOT
from mp135.modeling import predict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    hour: int = Field(..., ge=0, le=23)
    nearby_offices: int = Field(..., ge=0)
    temp_c: float
    rain_mm: float = Field(..., ge=0, le=100)
    feeder_buses: int = Field(..., ge=0)
    ticket_price_rs: float = Field(..., ge=5)
    day_type: Literal["weekday", "weekend", "holiday"]
    line: Literal["Red Line", "Blue Line", "Yellow Line"]


class PredictionResponse(BaseModel):
    prediction: float = Field(..., ge=0)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "event_data", {}),
        }, allow_nan=False)


logger = logging.getLogger("mp135.predictions")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def create_app(model=None, model_path=None):
    @asynccontextmanager
    async def lifespan(application):
        application.state.model = model if model is not None else joblib.load(
            model_path or ROOT / "models" / "model.joblib",
        )
        yield

    application = FastAPI(title="MP135 Metro Crowd API", version="1.0", lifespan=lifespan)

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.post("/predict", response_model=PredictionResponse)
    def prediction(request: PredictionRequest):
        started = time.perf_counter()
        frame = pd.DataFrame([request.model_dump()])
        result = float(predict(application.state.model, frame)[0])
        if not math.isfinite(result):
            raise HTTPException(status_code=500, detail="Model returned a nonfinite prediction")
        logger.info("prediction", extra={"event_data": {
            "prediction": result, "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }})
        return {"prediction": result}

    return application


app = create_app()
