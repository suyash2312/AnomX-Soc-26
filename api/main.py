# api/main.py
#
# FastAPI application for the ForexGuard anomaly detection engine.
#
# Endpoints:
#   GET  /health  — check if server and model are ready
#   POST /score   — score a single event and return anomaly verdict
#
# Run with:
#   uvicorn api.main:app --reload
#
# Docs available at:
#   http://localhost:8000/docs

import sys
from pathlib import Path

# make sure project root is on the path so imports work
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import EventRequest, ScoreResponse, HealthResponse, FeatureDetail
from models.scorer import ForexGuardScorer

# ── App setup ─────────────────────────────────────────────────────────────────

# single global scorer instance
# model is loaded once at startup, not on every request
scorer = ForexGuardScorer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when the server starts up."""
    print("[startup] Loading ForexGuard model...")
    scorer.load()
    if scorer.loaded:
        print("[startup] Model ready.")
    else:
        print("[startup] WARNING: Model failed to load. Run isolation_forest.py first.")
    yield
    print("[shutdown] Server shutting down.")


app = FastAPI(
    title="ForexGuard — AnomX Anomaly Detection API",
    description=(
        "Real-time anomaly detection for financial trading activity. "
        "Detects brute force logins, IP hopping, wash trading, structuring, "
        "bot activity, dormant account takeovers, and deposit-withdrawal cycling."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns whether the server is up and the model is loaded."
)
def health():
    return {
        "status":       "ok",
        "model_loaded": scorer.loaded,
    }


# ── POST /score ───────────────────────────────────────────────────────────────

@app.post(
    "/score",
    response_model=ScoreResponse,
    summary="Score an event",
    description=(
        "Accepts a single financial event with all engineered features "
        "and returns an anomaly score, severity, human-readable reasons, "
        "and the top contributing features."
    ),
    responses={
        503: {"description": "Model not loaded"},
        422: {"description": "Invalid request body"},
    }
)
def score_event(event: EventRequest):
    # return 503 if model isn't loaded yet
    if not scorer.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python models/isolation_forest.py first."
        )

    # convert pydantic model to dict for scorer
    event_dict = event.model_dump()

    try:
        result = scorer.score(event_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

    # build response
    return ScoreResponse(
        user_id       = result["user_id"],
        event_type    = result["event_type"],
        anomaly_score = result["anomaly_score"],
        is_anomaly    = result["is_anomaly"],
        severity      = result["severity"],
        verdict       = result["verdict"],
        reasons       = result["reasons"],
        top_features  = [FeatureDetail(**f) for f in result["top_features"]],
    )
