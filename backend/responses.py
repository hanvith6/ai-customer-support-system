"""
Response generation module.

Combines intent classification, NER extraction, and sentiment analysis
to produce the final response payload for the /chat endpoint.
"""

from backend.intent_model import classify_intent, get_intent_response
from backend.ner_model import extract_entities
from backend.sentiment import analyze_sentiment
from backend.logging_config import logger

FALLBACK_RESPONSE = (
    "I'm sorry, I didn't quite understand that. "
    "Could you please rephrase your question?"
)


def generate_response(text: str) -> dict:
    """
    Process a user message through the full NLP pipeline.

    Returns:
        Dict with keys: intent, confidence, entities, sentiment, response.
    """
    logger.info("Pipeline started | input=%r", text[:100])

    # 1. Intent classification
    intent, confidence = classify_intent(text)
    logger.info("Intent predicted | tag=%s | confidence=%.4f", intent, confidence)

    # 2. Entity extraction
    entities = extract_entities(text)
    entity_summary = [f"{e['word']}({e['entity']})" for e in entities]
    logger.info("Entities extracted | count=%d | entities=%s", len(entities), entity_summary)

    # 3. Sentiment analysis
    sentiment = analyze_sentiment(text)
    logger.info("Sentiment analyzed | label=%s | score=%.4f", sentiment["label"], sentiment["score"])

    # 4. Build response
    if intent == "unknown":
        response_text = FALLBACK_RESPONSE
    else:
        response_text = get_intent_response(intent)

    logger.info("Response generated | intent=%s | response=%r", intent, response_text[:80])

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "entities": entities,
        "sentiment": sentiment,
        "response": response_text,
    }
