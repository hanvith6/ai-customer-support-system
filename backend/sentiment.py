"""
Sentiment analysis module (extra integration).

Provides a lightweight sentiment classifier using a HuggingFace
distilbert pipeline. Enriches chat responses with user sentiment
detection so the system can adapt its tone.
"""

from transformers import pipeline

from backend.logging_config import logger

_sentiment_pipeline = None


def _load_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading sentiment analysis model")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
    return _sentiment_pipeline


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a user message.

    Returns:
        Dict with keys:
        - ``label``: POSITIVE or NEGATIVE
        - ``score``: confidence score (0-1)
    """
    try:
        pipe = _load_pipeline()
        result = pipe(text, truncation=True, max_length=512)[0]
        return {
            "label": result["label"],
            "score": round(float(result["score"]), 4),
        }
    except Exception as exc:
        logger.error("Sentiment analysis failed: %s", exc)
        return {"label": "UNKNOWN", "score": 0.0}
