# Architecture

## Overview

The AI Customer Support System processes customer messages through three AI models -- intent classification, named entity recognition, and sentiment analysis -- then returns a structured JSON response through a FastAPI REST API.

---

## System Diagram

```
                        ┌────────────────┐
                        │      User      │
                        └───────┬────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Frontend Chat UI   │
                     │   (HTML / CSS / JS)  │
                     │                     │
                     │  - Floating widget   │
                     │  - Quick-reply chips │
                     │  - NER/sentiment     │
                     │    metadata pills    │
                     └──────────┬──────────┘
                                │  POST /chat
                                ▼
                     ┌─────────────────────┐
                     │   FastAPI Backend    │
                     │                     │
                     │  - Pydantic input    │
                     │    validation        │
                     │  - Sliding-window    │
                     │    rate limiter      │
                     │  - CORS middleware   │
                     │  - Structured logs   │
                     └──────────┬──────────┘
                                │
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
          ┌────────────┐ ┌───────────┐ ┌────────────┐
          │  Intent    │ │    NER    │ │ Sentiment  │
          │ Classifier │ │ Extractor │ │  Analyzer  │
          │            │ │           │ │            │
          │ PyTorch NN │ │ BERT NER  │ │ DistilBERT │
          │ (3-layer)  │ │ pipeline  │ │ pipeline   │
          └─────┬──────┘ └─────┬─────┘ └─────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │  Response Generator  │
                    │                     │
                    │  intent tag →       │
                    │  intents.json       │
                    │  lookup + entities  │
                    │  + sentiment        │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   JSON API Response  │
                    │                     │
                    │  {                  │
                    │    "intent": "...", │
                    │    "confidence": N, │
                    │    "entities": [...],│
                    │    "sentiment": {}, │
                    │    "response": "..."│
                    │  }                  │
                    └─────────────────────┘
```

---

## AI Pipeline

The NLP pipeline (`backend/responses.py`) orchestrates three models on every request:

### 1. Intent Classifier

**File:** `backend/intent_model.py`

| Aspect | Detail |
|--------|--------|
| Architecture | 3-layer feed-forward NN: Linear - ReLU - Linear - ReLU - Linear |
| Input | Bag-of-words vector from NLTK tokenization + Porter stemming |
| Training data | `model/intents.json` -- 14 customer-support intent categories |
| Pretrained weights | `model/data.pth` -- loads on startup; trains automatically if missing |
| Confidence cutoff | 0.75 -- below this the intent is returned as `unknown` |

**Processing flow:**

```
User text → NLTK tokenize → Porter stem → bag-of-words vector → PyTorch NN → (intent_tag, confidence)
```

### 2. NER Extractor

**File:** `backend/ner_model.py`

| Aspect | Detail |
|--------|--------|
| Model | `dslim/bert-base-NER` (BERT fine-tuned for token classification) |
| Framework | HuggingFace `transformers.pipeline("ner")` |
| Entity types | PER (person), ORG (organization), LOC (location), MISC |
| Fallback | Loads from `model/ner_model/` if a local model exists |

**Processing flow:**

```
User text → BERT tokenizer → token classification → entity spans with labels and scores
```

### 3. Sentiment Analyzer

**File:** `backend/sentiment.py`

| Aspect | Detail |
|--------|--------|
| Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Output | POSITIVE or NEGATIVE with confidence score |
| Purpose | Detects user frustration so the system can adapt tone |
| Loading | Lazy-loaded on first request to speed up startup |

**Processing flow:**

```
User text → DistilBERT tokenizer → sequence classification → (label, score)
```

---

## Model Components

### Preprocessing (`backend/preprocess.py`)

Three utility functions for the intent classifier input pipeline:

- **`tokenize(sentence)`** -- splits text into word tokens via NLTK.
- **`stem(word)`** -- Porter stemmer to find root form, lowercased.
- **`bag_of_words(tokens, vocabulary)`** -- binary feature vector for the classifier.

### Response Schema (`backend/response_schema.py`)

Pydantic models that enforce a strict JSON contract for every API response:

- `ChatRequestSchema` -- validates incoming user messages (1--1000 chars).
- `ChatResponseSchema` -- validates the full pipeline output before returning.
- `EntitySchema` / `SentimentSchema` -- nested sub-schemas for type safety.

### Configuration (`backend/config.py`)

All paths, thresholds, and settings are centralized:

- Model paths resolved relative to the project root via `pathlib`.
- Intent confidence threshold, NER model name, rate limit parameters.
- API host/port and CORS origins.

---

## Data Flow

1. User types a message in the chat widget or clicks a quick-reply button.
2. Frontend sends `POST /chat` with `{ "text": "..." }` via `fetch()`.
3. FastAPI validates the input schema, enforces rate limit, checks body size.
4. The NLP pipeline runs three models:
   - **Intent classifier** -- tokenize, stem, bag-of-words, PyTorch NN inference.
   - **NER extractor** -- BERT token classification, entity span extraction.
   - **Sentiment analyzer** -- DistilBERT sequence classification.
5. Response generator looks up a reply in `intents.json` for the classified intent.
6. The full result is validated against `ChatResponseSchema` and returned as JSON.
7. Frontend renders the reply bubble with intent/entity/sentiment metadata pills.

---

## API Layer (`backend/app.py`)

| Aspect | Detail |
|--------|--------|
| Framework | FastAPI with Uvicorn ASGI server |
| Endpoints | `POST /chat`, `GET /health`, `GET /` (frontend) |
| Validation | Pydantic schemas, max 1000 characters |
| Rate limiting | In-memory sliding window, 60 requests/min per IP |
| Request size | Capped at 1024 bytes |
| CORS | Configurable origins (default: all) |
| Logging | Structured logs for every request, prediction, and error |

---

## Frontend (`frontend/`)

A lightweight vanilla HTML/CSS/JS chat interface served as static files by FastAPI.

- **Floating widget toggle** -- corner icon to open/close the chat window.
- **Quick-reply buttons** -- pre-filled chips for common queries (Billing Issue, Track Order, Tech Support, View Plans).
- **NLP metadata pills** -- colored badges showing detected intent, sentiment, and entities.
- **Typing indicator** -- animated dots while waiting for the API response.

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
│   ├── script.js            # Widget toggle, quick replies, API calls
│   └── style.css            # Responsive styles
├── tests/
│   └── test_api.py          # Pytest API test suite (9 tests)
├── docs/
│   └── architecture.md      # This document
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── .gitignore
└── README.md
```
