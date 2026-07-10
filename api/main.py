import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from api.schemas import EventRequest, ScoreResponse, HealthResponse, FeatureDetail
from models.scorer import ForexGuardScorer

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


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns whether the server is up and the model is loaded.",
)
def health():
    return {
        "status": "ok",
        "model_loaded": scorer.loaded,
    }


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
    },
)
def score_event(event: EventRequest):
    if not scorer.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python models/isolation_forest.py first.",
        )

    event_dict = event.model_dump()

    try:
        result = scorer.score(event_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

    return ScoreResponse(
        user_id=result["user_id"],
        event_type=result["event_type"],
        anomaly_score=result["anomaly_score"],
        is_anomaly=result["is_anomaly"],
        severity=result["severity"],
        verdict=result["verdict"],
        reasons=result["reasons"],
        top_features=[FeatureDetail(**f) for f in result["top_features"]],
    )
