"""Embedders for dense retrieval.

The default, :class:`HashingEmbedder`, needs no model download: it is a TF-IDF
vector over hashed word unigrams and character n-grams. That is not a semantic
model, but it is a genuinely different *representation* from BM25: character
n-grams match "replicated" to "replication" and "consumers" to "consumer", which
whole-word BM25 cannot. That gives the hybrid benchmark a real complementary signal
offline. Swap in :class:`SentenceTransformerEmbedder` (``pip install
'ragscope[st]'``) for actual semantic embeddings; the rest of the pipeline is
unchanged because both satisfy the same protocol.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .text import tokenize


@runtime_checkable
class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None:
        """Learn corpus statistics (e.g. idf). Stateless embedders make this a no-op."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalised rows."""


class HashingEmbedder:
    """Signed feature hashing of words + char n-grams, sublinear TF, bucket IDF.

    Design choices:
    * ``zlib.crc32`` rather than ``hash()``: Python's string hash is salted per
      process (PYTHONHASHSEED), which would make vectors differ between runs.
    * Signed hashing (one bit of a second hash picks +1/-1) makes collisions cancel
      in expectation instead of always inflating similarity.
    * IDF is computed per bucket after hashing, as scikit-learn's
      HashingVectorizer + TfidfTransformer does. It keeps memory O(dim) regardless
      of vocabulary size.
    """

    name = "hashing"

    def __init__(self, dim: int = 4096, ngram_range: tuple[int, int] = (3, 5), char_weight: float = 0.5) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        lo, hi = ngram_range
        if not 1 <= lo <= hi:
            raise ValueError("ngram_range must satisfy 1 <= lo <= hi")
        self.dim = dim
        self.ngram_range = ngram_range
        self.char_weight = char_weight
        self.idf = np.ones(dim, dtype=np.float32)

    def _features(self, text: str) -> dict[int, float]:
        feats: dict[int, float] = {}
        lo, hi = self.ngram_range
        for tok in tokenize(text):
            self._add(feats, "w:" + tok, 1.0)
            padded = f"<{tok}>"  # boundary markers make prefixes/suffixes distinctive
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    self._add(feats, padded[i : i + n], self.char_weight)
        return feats

    def _add(self, feats: dict[int, float], feature: str, weight: float) -> None:
        h = zlib.crc32(feature.encode())
        idx = h % self.dim
        sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
        feats[idx] = feats.get(idx, 0.0) + sign * weight

    def fit(self, texts: Sequence[str]) -> None:
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            df[list(self._features(text))] += 1
        n = len(texts)
        # Smoothed idf (as in scikit-learn): never zero, never divides by zero.
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for idx, val in self._features(text).items():
                # Sublinear tf on the magnitude, keep the sign from hashing.
                out[row, idx] = math.copysign(1 + math.log(abs(val)), val) if abs(val) >= 1 else val
        out *= self.idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # an empty text stays the zero vector, not NaN
        return out / norms


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Optional extra: ``ragscope[st]``."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "SentenceTransformerEmbedder needs the optional extra: pip install 'ragscope[st]'"
            ) from exc
        self.model_name = model
        self._model = SentenceTransformer(model)

    def fit(self, texts: Sequence[str]) -> None:
        return None  # pretrained; nothing to learn from the corpus

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - optional
        vecs = self._model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def make_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name in ("st", "sentence-transformers"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {name!r}; choose 'hashing' or 'st'")
