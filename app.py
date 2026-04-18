"""
ML Inference API
FastAPI-based production inference server for sklearn models.
"""

import time
import logging
import pickle
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ml_inference")

# ─── Model Store ────────────────────────────────────────────────────────────

MODEL_PATH = Path("model.pkl")

model_store: dict[str, Any] = {
    "model": None,
    "feature_names": [],
    "target_names": [],
    "description": "",
    "loaded": False,
    "loaded_at": None,
}
request_count = 0


def load_model() -> None:
    """Load the pickled model into memory once at startup."""
    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)

        model_store["model"] = payload["model"]
        model_store["feature_names"] = payload.get("feature_names", [])
        model_store["target_names"] = payload.get("target_names", [])
        model_store["description"] = payload.get("description", "Unknown model")
        model_store["loaded"] = True
        model_store["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(f"Model loaded: {model_store['description']}")
        logger.info(f"Features: {model_store['feature_names']}")
        logger.info(f"Classes: {model_store['target_names']}")
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ML Inference API …")
    load_model()
    yield
    logger.info("Shutting down ML Inference API.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Inference API",
    description="Production-grade inference server for scikit-learn models",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ─── Schemas ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    @field_validator("sepal_length", "sepal_width", "petal_length", "petal_width")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("must be a positive number")
        if v > 100:
            raise ValueError(f"value {v} seems unrealistically large")
        return v


class PredictResponse(BaseModel):
    prediction: str
    prediction_index: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_description: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_description: str
    model_loaded_at: str | None
    feature_names: list[str]
    uptime_note: str


# ─── Middleware ───────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"({duration_ms:.1f}ms)"
    )
    return response


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the frontend UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return system and model health status."""
    return HealthResponse(
        status="ok",
        model_loaded=model_store["loaded"],
        model_description=model_store["description"] or "No model loaded",
        model_loaded_at=model_store["loaded_at"],
        feature_names=model_store["feature_names"],
        uptime_note="Service is running normally",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    global request_count
    request_count += 1
    """Run inference and return prediction with confidence and latency."""
    if not model_store["loaded"]:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    features = [
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]

    t0 = time.perf_counter()

    try:
        # Run CPU-bound inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        pred_idx, proba = await loop.run_in_executor(
            None, _run_inference, features
        )
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    latency_ms = (time.perf_counter() - t0) * 1000
    target_names = model_store["target_names"]
    pred_label = target_names[pred_idx] if target_names else str(pred_idx)

    probabilities = {
        name: round(float(p), 4)
        for name, p in zip(target_names, proba)
    }

    logger.info(
        f"Prediction: {pred_label} (confidence={proba[pred_idx]:.2%}) "
        f"latency={latency_ms:.2f}ms"
    )

    return {
    "prediction": pred_label,
    "prediction_index": int(pred_idx),
    "confidence": round(float(proba[pred_idx]), 4),
    "probabilities": probabilities,
    "latency_ms": round(latency_ms, 3),
    "model_description": model_store["description"],
    "requests_served": request_count
}


def _run_inference(features: list[float]) -> tuple[int, list[float]]:
    """Synchronous inference helper (executed in thread pool)."""
    import numpy as np
    model = model_store["model"]
    X = np.array([features])
    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    return pred_idx, proba


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]
    messages = [f"{e['loc'][-1]}: {e['msg']}" for e in errors if e.get("loc")]
    return JSONResponse(
        status_code=422,
        content={"detail": messages or ["Validation error"], "type": "validation_error"},
    )
