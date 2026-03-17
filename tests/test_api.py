"""
Tests for the AI Customer Support System API.

Run with:  pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


# ── Health check ───────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "AI Customer Support System"


# ── POST /chat — valid requests ────────────────────────────────────────

def test_chat_greeting():
    response = client.post("/chat", json={"text": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "response" in data
    assert "entities" in data
    assert "sentiment" in data
    assert isinstance(data["entities"], list)
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_chat_billing():
    response = client.post("/chat", json={"text": "I have a billing issue"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] in ("billing", "unknown")
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_chat_returns_sentiment():
    response = client.post("/chat", json={"text": "Thank you so much!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"]["label"] in ("POSITIVE", "NEGATIVE", "UNKNOWN")
    assert 0.0 <= data["sentiment"]["score"] <= 1.0


def test_chat_returns_entities():
    response = client.post("/chat", json={"text": "My name is John and I live in New York"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["entities"], list)
    if len(data["entities"]) > 0:
        ent = data["entities"][0]
        assert "entity" in ent
        assert "word" in ent
        assert "score" in ent


def test_chat_unknown_intent():
    response = client.post("/chat", json={"text": "xyzzy foobar baz"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["response"], str)


# ── POST /chat — validation errors ────────────────────────────────────

def test_chat_empty_text():
    response = client.post("/chat", json={"text": ""})
    assert response.status_code == 422


def test_chat_missing_text():
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_text_too_long():
    long_text = "a" * 1001
    response = client.post("/chat", json={"text": long_text})
    assert response.status_code == 422


# ── GET / ──────────────────────────────────────────────────────────────

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
