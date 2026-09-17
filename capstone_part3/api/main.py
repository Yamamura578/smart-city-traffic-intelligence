"""Part 3, Task 6.3 — FastAPI deployment mock-up.

Run from the repository root:

    uvicorn capstone_part3.api.main:app --reload
    # then open http://127.0.0.1:8000/docs

Serves two trained models:
  POST /predict/volume  → expected traffic volume for stated conditions
  POST /predict/risk    → proxy accident-risk probability
  GET  /recommend       → travel-timing recommendation
  GET  /health          → model and service status

This is a simulation of deployment, not a production service: there is no
authentication, rate limiting, input provenance checking or model registry
lookup. Those gaps are discussed in the final report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from common.config import PART3_MODELS
from common.logging_config import get_logger
from capstone_part3.src.data import FEATURE_COLUMNS
from capstone_part3.src.recommender import recommend

logger = get_logger(__name__)

app = FastAPI(
    title="Smart City Traffic Intelligence API",
    description="Deployment simulation for the NUS SOC AI/ML/DS capstone.",
    version="1.0.0",
)

MODELS: dict = {}
WEATHER_CATEGORIES = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
                      "Other", "Rain", "Snow", "Thunderstorm"]
SEVERE = {"Rain", "Snow", "Thunderstorm", "Squall"}
LOW_VIS = {"Fog", "Mist", "Haze", "Smoke"}


@app.on_event("startup")
def load_models() -> None:
    for key, filename in (
        ("regressor", "regressor_gradient_boosting.joblib"),
        ("classifier", "classifier_random_forest.joblib"),
    ):
        path = PART3_MODELS / filename
        try:
            MODELS[key] = joblib.load(path)
            logger.info("Model loaded | %s | path=%s", key, path)
        except FileNotFoundError:
            logger.error("Model not found | %s | path=%s — run supervised.py first",
                         key, path, exc_info=True)


class Conditions(BaseModel):
    """Inputs a caller would realistically have at prediction time."""
    timestamp: datetime = Field(..., examples=["2018-06-15T08:00:00"])
    weather: Literal[tuple(WEATHER_CATEGORIES)] = Field("Clear", examples=["Rain"])
    temp_celsius: float = Field(20.0, ge=-45, le=50)
    clouds_all: int = Field(40, ge=0, le=100)
    rain_1h: float = Field(0.0, ge=0, le=100)
    snow_1h: float = Field(0.0, ge=0, le=100)
    is_holiday: bool = False


def build_feature_row(c: Conditions) -> pd.DataFrame:
    """Assemble the 27-column feature vector the models were trained on."""
    ts = c.timestamp
    row = {
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
        "is_holiday": int(c.is_holiday),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin": np.sin(2 * np.pi * ts.weekday() / 7),
        "dow_cos": np.cos(2 * np.pi * ts.weekday() / 7),
        "temp": c.temp_celsius + 273.15,
        "rain_1h": c.rain_1h,
        "snow_1h": c.snow_1h,
        "clouds_all": c.clouds_all,
        "is_precipitation": int(c.weather in SEVERE),
        "is_low_visibility": int(c.weather in LOW_VIS),
        "has_rain": int(c.rain_1h > 0),
        "has_snow": int(c.snow_1h > 0),
    }
    for category in WEATHER_CATEGORIES:
        row[f"weather_{category}"] = int(c.weather == category)
    return pd.DataFrame([row])[FEATURE_COLUMNS].astype(float)


@app.get("/health")
def health() -> dict:
    status = {k: (k in MODELS) for k in ("regressor", "classifier")}
    ready = all(status.values())
    logger.info("Health check | ready=%s | models=%s", ready, status)
    return {"status": "ok" if ready else "degraded", "models_loaded": status}


@app.post("/predict/volume")
def predict_volume(conditions: Conditions) -> dict:
    if "regressor" not in MODELS:
        logger.error("Volume prediction requested but regressor is not loaded")
        raise HTTPException(status_code=503, detail="Regression model unavailable")

    features = build_feature_row(conditions)
    prediction = float(MODELS["regressor"].predict(features)[0])
    logger.info("Volume prediction | timestamp=%s | weather=%s | predicted=%.0f",
                conditions.timestamp, conditions.weather, prediction)
    return {
        "predicted_volume": round(prediction, 1),
        "units": "vehicles per hour",
        "model": "gradient_boosting",
        "note": "Point estimate; the model's test MAE is approximately 230 vehicles/hour.",
    }


@app.post("/predict/risk")
def predict_risk(conditions: Conditions) -> dict:
    if "classifier" not in MODELS:
        logger.error("Risk prediction requested but classifier is not loaded")
        raise HTTPException(status_code=503, detail="Classification model unavailable")

    features = build_feature_row(conditions)
    probability = float(MODELS["classifier"].predict_proba(features)[0, 1])
    logger.info("Risk prediction | timestamp=%s | weather=%s | probability=%.3f",
                conditions.timestamp, conditions.weather, probability)
    return {
        "high_risk_probability": round(probability, 4),
        "label": int(probability >= 0.5),
        "model": "random_forest",
        "warning": (
            "This is a PROXY risk score derived from congestion and weather. "
            "No accident data was used. It must not be read as an accident "
            "probability or used for safety-critical decisions."
        ),
    }


@app.get("/recommend")
def recommend_travel(
    day_type: Literal["all", "weekday", "weekend"] = "weekday",
    weather: str | None = None,
    earliest: int = 6,
    latest: int = 20,
) -> dict:
    if not (0 <= earliest <= 23 and 0 <= latest <= 23 and earliest <= latest):
        logger.error("Invalid hour window requested | earliest=%d latest=%d",
                     earliest, latest)
        raise HTTPException(status_code=422, detail="Invalid hour window")
    logger.info("Recommendation requested | day_type=%s weather=%s window=%d-%d",
                day_type, weather, earliest, latest)
    return recommend(day_type=day_type, weather=weather,
                     earliest=earliest, latest=latest)
