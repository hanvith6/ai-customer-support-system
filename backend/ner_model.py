"""
Named Entity Recognition module.

Uses a HuggingFace transformers NER pipeline. Falls back to a lightweight
pretrained model (dslim/bert-base-NER) that runs without a GPU.

Adapted from the Chatbot_NLP_KJIT_telecom source project NER components.
"""

from transformers import pipeline

from backend.config import NER_MODEL_DIR, NER_MODEL_NAME
from backend.logging_config import logger


def _load_ner_pipeline():
    """Load NER pipeline from local model dir or download a pretrained one."""
    # Check for locally trained NER model from source project
    config_path = NER_MODEL_DIR / "config.json"
    if NER_MODEL_DIR.exists() and config_path.exists():
        try:
            logger.info("Loading local NER model from %s", NER_MODEL_DIR)
            return pipeline(
                "ner",
                model=str(NER_MODEL_DIR),
                tokenizer=str(NER_MODEL_DIR),
                aggregation_strategy="simple",
            )
        except Exception as exc:
            logger.warning("Failed to load local NER model: %s. Falling back to %s", exc, NER_MODEL_NAME)

    logger.info("Loading NER model: %s", NER_MODEL_NAME)
    return pipeline("ner", model=NER_MODEL_NAME, aggregation_strategy="simple")


_ner_pipeline = _load_ner_pipeline()


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from text.

    Returns:
        List of dicts, each with keys:
        - ``entity``: entity label (e.g. PER, ORG, LOC, MISC)
        - ``word``: the matched text span
        - ``score``: confidence score (0-1)
        - ``start``: character start offset
        - ``end``: character end offset
    """
    try:
        raw = _ner_pipeline(text)
    except Exception as exc:
        logger.error("NER extraction failed: %s", exc)
        return []

    entities = []
    for ent in raw:
        entities.append({
            "entity": ent.get("entity_group", ent.get("entity", "UNKNOWN")),
            "word": ent.get("word", ""),
            "score": round(float(ent.get("score", 0.0)), 4),
            "start": ent.get("start", 0),
            "end": ent.get("end", 0),
        })

    logger.info("Extracted %d entities from text", len(entities))
    return entities
