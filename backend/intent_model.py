"""
Intent classification model and inference.

Uses a PyTorch feed-forward neural network with bag-of-words input.
If a pretrained model (data.pth) exists it is loaded; otherwise the
model is trained from intents.json on first import.

Architecture adapted from the AI-Chatbot-DL-NLP source project.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from backend.preprocess import tokenize, stem, bag_of_words
from backend.config import INTENT_DATA_PATH, INTENT_MODEL_PATH, INTENT_CONFIDENCE_THRESHOLD
from backend.logging_config import logger


# ── Neural network ─────────────────────────────────────────────────────

class NeuralNet(nn.Module):
    """Three-layer feed-forward classifier."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        return self.l3(out)


# ── Training helpers ───────────────────────────────────────────────────

class _IntentDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def _train_model(intents_path: Path, save_path: Path) -> dict:
    """Train the intent classifier from intents.json and save to data.pth."""
    logger.info("Training intent model from %s", intents_path)

    with open(intents_path, "r") as f:
        intents = json.load(f)

    all_words: list[str] = []
    tags: list[str] = []
    xy: list[tuple[list[str], str]] = []
    ignore_chars = {"?", ".", "!", ","}

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            words = tokenize(pattern)
            all_words.extend(words)
            xy.append((words, tag))

    all_words = sorted(set(stem(w) for w in all_words if w not in ignore_chars))
    tags = sorted(set(tags))

    X_train = np.array([bag_of_words(pat, all_words) for pat, _ in xy])
    y_train = np.array([tags.index(tag) for _, tag in xy])

    # Hyperparameters
    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    hidden_size = 8
    input_size = len(X_train[0])
    output_size = len(tags)

    dataset = _IntentDataset(X_train, y_train)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for words, labels in loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long, device=device)
            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 200 == 0:
            logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, num_epochs, loss.item())

    logger.info("Training complete. Final loss: %.4f", loss.item())

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, save_path)
    logger.info("Model saved to %s", save_path)
    return data


# ── Model loading ──────────────────────────────────────────────────────

def _load_model_data() -> dict:
    """Load pretrained model or train a new one."""
    if INTENT_MODEL_PATH.exists():
        logger.info("Loading pretrained intent model from %s", INTENT_MODEL_PATH)
        return torch.load(INTENT_MODEL_PATH, map_location="cpu", weights_only=False)
    return _train_model(INTENT_DATA_PATH, INTENT_MODEL_PATH)


_data = _load_model_data()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = NeuralNet(
    _data["input_size"], _data["hidden_size"], _data["output_size"]
).to(_device)
_model.load_state_dict(_data["model_state"])
_model.eval()

_all_words: list[str] = _data["all_words"]
_tags: list[str] = _data["tags"]

# Load intents for response lookup
with open(INTENT_DATA_PATH, "r") as _f:
    _intents = json.load(_f)


# ── Public API ─────────────────────────────────────────────────────────

def classify_intent(text: str) -> tuple[str, float]:
    """
    Classify the intent of a user message.

    Returns:
        Tuple of (intent_tag, confidence). If confidence is below
        the threshold, intent_tag is ``"unknown"``.
    """
    tokens = tokenize(text)
    X = bag_of_words(tokens, _all_words)
    X = torch.from_numpy(X.reshape(1, -1)).to(_device)

    with torch.no_grad():
        output = _model(X)

    probs = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    confidence_val = confidence.item()
    tag = _tags[predicted.item()]

    if confidence_val < INTENT_CONFIDENCE_THRESHOLD:
        return "unknown", confidence_val

    return tag, confidence_val


def get_intent_response(tag: str) -> str:
    """Return a random response string for the given intent tag."""
    for intent in _intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to help with that. Could you rephrase your question?"
