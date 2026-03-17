<div align="center">

# AI Customer Support System

An AI-powered customer support platform that routes user messages through a **PyTorch intent classifier**, **transformer-based NER extraction**, and **DistilBERT sentiment analysis** — returning structured JSON responses via a FastAPI REST API with a live chat frontend.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Architecture

```
                        ┌────────────────┐
                        │      User      │
                        └───────┬────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Frontend Chat UI   │
                     │   (HTML / CSS / JS)  │
                     └──────────┬──────────┘
                                │  POST /chat
                                ▼
                     ┌─────────────────────┐
                     │   FastAPI API Layer  │
                     │   Validation, CORS,  │
                     │   Rate Limiting      │
                     └──────────┬──────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
          ┌────────────┐ ┌───────────┐ ┌────────────┐
          │  Intent    │ │    NER    │ │ Sentiment  │
          │ Classifier │ │ Extractor │ │  Analyzer  │
          │ (PyTorch)  │ │  (BERT)   │ │(DistilBERT)│
          └─────┬──────┘ └─────┬─────┘ └─────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Response Generator  │
                    │  → Structured JSON   │
                    └─────────────────────┘
```

| Component | Role |
|-----------|------|
| **Frontend Chat UI** | Floating chat widget with quick-reply buttons and NLP metadata pills |
| **FastAPI API Layer** | REST API with Pydantic validation, sliding-window rate limiting, CORS |
| **Intent Classifier** | 3-layer feed-forward neural network (bag-of-words input, 14 intent categories) |
| **NER Extractor** | HuggingFace `dslim/bert-base-NER` — extracts PER, ORG, LOC, MISC entities |
| **Sentiment Analyzer** | DistilBERT — classifies user sentiment as POSITIVE or NEGATIVE |
| **Response Generator** | Combines all model outputs into a validated JSON response |

See [`docs/architecture.md`](docs/architecture.md) for the full component breakdown.

---

## Features

| Feature | Description |
|---------|-------------|
| **Intent Classification** | PyTorch neural network trained on 14 customer-support intent categories |
| **Named Entity Recognition** | Transformer-based NER using `dslim/bert-base-NER` (PER, ORG, LOC, MISC) |
| **Sentiment Analysis** | DistilBERT pipeline detecting POSITIVE / NEGATIVE user sentiment |
| **FastAPI Backend** | REST API with Pydantic v2 schemas and automatic OpenAPI docs |
| **Rate Limiting** | In-memory sliding window — 60 requests/min per IP |
| **Structured Logging** | Pipeline-stage logging for every request, prediction, and error |
| **Health Monitoring** | `GET /health` endpoint for uptime checks |
| **Chat Frontend** | Floating widget with quick-reply buttons and NER/sentiment pills |
| **Auto-Training** | Intent model trains automatically from `intents.json` if weights are missing |
| **Test Suite** | 9 pytest tests covering endpoints, validation, and edge cases |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **ML / NLP** | PyTorch, HuggingFace Transformers, NLTK |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **Validation** | Pydantic v2 |
| **Testing** | pytest, FastAPI TestClient |

---

## Live Demo

Once deployed (or running locally), open these URLs in your browser:

| URL | Description |
|-----|-------------|
| `/` | Full-page demo UI -- interactive chat with intent, entity, and sentiment display |
| `/widget` | Floating chat widget |
| `/docs` | Interactive Swagger API documentation |
| `/health` | JSON health check |

---

## Quick Demo

Start the server, then run these commands:

**Chat request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "My internet is not working"}'
```

**Expected response:**
```json
{
  "intent": "technical_support",
  "confidence": 0.8734,
  "entities": [],
  "sentiment": {
    "label": "NEGATIVE",
    "score": 0.9821
  },
  "response": "I'm sorry to hear you're having technical issues. Can you describe the problem in more detail?"
}
```

**Entity extraction:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is John and I live in New York"}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```
```json
{ "status": "ok", "service": "AI Customer Support System" }
```

**Test on a deployed URL:**
```bash
curl -X POST https://your-deployment-url.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "I want a refund for my last order"}'
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hanvith6/ai-customer-support-system.git
cd ai-customer-support-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

NLTK data downloads automatically on first run. To pre-download:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Running Locally

```bash
python -m backend.app
```

The server starts at **http://localhost:8000**.

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Full-page demo UI |
| `http://localhost:8000/widget` | Floating chat widget |
| `http://localhost:8000/docs` | Interactive API docs (Swagger) |
| `http://localhost:8000/health` | Health check |

---

## Running Tests

```bash
pytest tests/test_api.py -v
```

---

## API Reference

### `GET /health`

Returns service status.

```json
{ "status": "ok", "service": "AI Customer Support System" }
```

### `POST /chat`

Process a customer message through the NLP pipeline.

**Request body:**

| Field | Type | Constraints |
|-------|------|-------------|
| `text` | `string` | 1 -- 1000 characters |

**Response body:**

| Field | Type | Description |
|-------|------|-------------|
| `intent` | `string` | Classified intent tag (e.g. `billing`, `greeting`, `unknown`) |
| `confidence` | `float` | Intent confidence score (0.0 -- 1.0) |
| `entities` | `array` | Extracted named entities with label, word, score, offsets |
| `sentiment` | `object` | `{ "label": "POSITIVE" \| "NEGATIVE", "score": float }` |
| `response` | `string` | Generated bot reply |

---

## Project Structure

```
ai-customer-support-system/
├── backend/
│   ├── app.py               # FastAPI application and endpoints
│   ├── intent_model.py      # PyTorch intent classifier + auto-training
│   ├── ner_model.py         # HuggingFace NER pipeline
│   ├── sentiment.py         # DistilBERT sentiment analysis
│   ├── preprocess.py        # Tokenization, stemming, bag-of-words
│   ├── responses.py         # NLP pipeline orchestrator
│   ├── response_schema.py   # Pydantic request / response schemas
│   ├── config.py            # Central configuration (paths, thresholds)
│   └── logging_config.py    # Structured logging setup
├── model/
│   ├── intents.json         # Intent definitions and response templates
│   ├── data.pth             # Pretrained intent model weights
│   └── ner_model/           # Optional local NER model
├── frontend/
│   ├── index.html           # Chat widget UI
│   ├── demo.html            # Full-page demo interface
│   ├── script.js            # Widget toggle, quick replies, API calls
│   └── style.css            # Responsive styles
├── tests/
│   └── test_api.py          # Pytest API test suite (9 tests)
├── docs/
│   └── architecture.md      # Architecture documentation
├── requirements.txt
├── render.yaml
├── LICENSE
├── CONTRIBUTING.md
├── .gitignore
└── README.md
```

---

## Training the Intent Model

A pretrained model (`model/data.pth`) is included. To retrain:

1. Edit `model/intents.json` to add or modify intent categories.
2. Delete `model/data.pth`.
3. Start the server -- it trains automatically on first launch (1000 epochs, a few seconds on CPU).

---

## Deployment

### Render

1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository.
3. Render auto-detects `render.yaml` -- or configure manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port 10000`
   - **Environment:** Python 3.10+
4. Deploy. Your demo is live at `https://your-service.onrender.com`.

### Railway

1. Create a new project on [Railway](https://railway.app).
2. Connect your GitHub repository.
3. Set the start command:
   ```
   python -m backend.app
   ```
4. Deploy. Railway assigns a public URL automatically.

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "backend.app"]
```

```bash
docker build -t ai-customer-support .
docker run -p 8000:8000 ai-customer-support
```

### Deployment Notes

- The app binds to `0.0.0.0:8000` (compatible with all cloud platforms).
- No external database required -- all models run in-memory.
- First startup downloads HuggingFace models (~500 MB). Subsequent starts use the local cache.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the [MIT License](LICENSE).
