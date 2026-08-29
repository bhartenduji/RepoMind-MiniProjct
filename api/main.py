from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.predict_bug_fix import load_model, predict


MODEL = None
TOKENIZER = None
CHECKPOINT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, CHECKPOINT

    MODEL, TOKENIZER, CHECKPOINT = load_model()

    yield

    MODEL = None
    TOKENIZER = None
    CHECKPOINT = None


app = FastAPI(
    title="RepoMind Bug-Fix Classifier API",
    version="1.0.0",
    description="API for predicting whether a code patch is a bug-fix.",
    lifespan=lifespan,
)


class PatchRequest(BaseModel):
    patch: str = Field(
        ...,
        description="Unified diff / patch text.",
    )

    added_line_count: int | None = None
    removed_line_count: int | None = None
    diff_stats: dict | None = None
    changed_files: list[dict] | None = None


class PredictionResponse(BaseModel):
    prediction: str
    non_bug_fix_probability: float
    bug_fix_probability: float


@app.get("/health")
def health():
    if MODEL is None:
        return {
            "status": "starting",
            "model": "not_loaded",
        }

    return {
        "status": "healthy",
        "model": "loaded",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_bug_fix(request: PatchRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    record = request.model_dump()

    try:
        result = predict(
            record,
            MODEL,
            TOKENIZER,
            CHECKPOINT,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return result
