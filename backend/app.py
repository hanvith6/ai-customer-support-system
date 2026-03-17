"""
FastAPI application for the AI Customer Support System.

Provides a REST API with:
- POST /chat   — process user messages through the NLP pipeline
- GET  /health — service health check
- GET  /       — serve the frontend chat UI
"""

import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import (
    CORS_ORIGINS,
    MAX_REQUEST_SIZE,
    BASE_DIR,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from backend.responses import generate_response
from backend.response_schema import ChatRequestSchema, ChatResponseSchema
from backend.logging_config import logger

# ── App setup ──────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Customer Support System",
    description="NLP-powered customer support chatbot API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_frontend_dir = BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


# ── Rate limiting (in-memory sliding window) ──────────────────────────

_request_log: dict[str, list[float]] = {}


def _check_rate_limit(client_ip: str) -> None:
    """Enforce per-IP sliding-window rate limit using config values."""
    now = time.time()
    timestamps = _request_log.get(client_ip, [])
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )
    timestamps.append(now)
    _request_log[client_ip] = timestamps


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend chat UI."""
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Customer Support System API", "docs": "/docs"}


@app.get("/demo")
async def demo():
    """Serve the standalone demo page for recruiter review."""
    demo_path = _frontend_dir / "demo.html"
    if demo_path.exists():
        return FileResponse(str(demo_path))
    return {"message": "Demo page not found. Ensure frontend/demo.html exists."}


@app.get("/health")
async def health():
    """Service health check."""
    return {
        "status": "ok",
        "service": "AI Customer Support System",
    }


@app.post("/chat", response_model=ChatResponseSchema)
async def chat(request: Request, body: ChatRequestSchema):
    """
    Process a user message through the NLP pipeline.

    Returns intent classification, named entities, sentiment, and a response.
    """
    client_ip = request.client.host if request.client else "unknown"

    # Rate limiting
    _check_rate_limit(client_ip)

    # Validate request size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        raise HTTPException(status_code=413, detail="Request body too large.")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text field cannot be empty.")

    logger.info("POST /chat | ip=%s | text=%r", client_ip, text[:80])

    try:
        result = generate_response(text)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error.")

    logger.info(
        "Response | intent=%s | confidence=%.2f | entities=%d | sentiment=%s",
        result["intent"],
        result["confidence"],
        len(result["entities"]),
        result["sentiment"]["label"],
    )
    return result


# ── Run directly ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from backend.config import API_HOST, API_PORT

    uvicorn.run("backend.app:app", host=API_HOST, port=API_PORT, reload=True)
