"""
NLP preprocessing utilities.

Provides tokenization, stemming, and bag-of-words conversion
for the intent classification pipeline.

Adapted from the AI-Chatbot-DL-NLP source project (nltk_utils.py).
"""

import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK data is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_stemmer = PorterStemmer()


def tokenize(sentence: str) -> list[str]:
    """Split a sentence into an array of word tokens."""
    return nltk.word_tokenize(sentence)


def stem(word: str) -> str:
    """Return the stemmed (root) form of a word, lowercased."""
    return _stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence: list[str], vocabulary: list[str]) -> np.ndarray:
    """
    Build a bag-of-words vector for a tokenized sentence given a vocabulary.

    Returns a float32 numpy array of length ``len(vocabulary)`` where index *i*
    is 1.0 if ``vocabulary[i]`` appears (after stemming) in the sentence,
    and 0.0 otherwise.
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, w in enumerate(vocabulary):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
