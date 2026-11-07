"""Okapi BM25, implemented from scratch over an inverted index.

score(q, d) = sum over terms t in q of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))   (the Lucene/BM25+ form)

The "+1" inside the log matters: the original Robertson-Sparck Jones idf goes
negative for terms in more than half the corpus, which would make matching a common
word *lower* a document's score. Small corpora hit that constantly.

k1 controls term-frequency saturation (the 5th occurrence of a word adds much less
than the 1st); b controls length normalisation (b=0 ignores length, b=1 fully
normalises). 1.5 / 0.75 are the textbook defaults and also what Lucene ships.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from .text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("require k1 >= 0 and 0 <= b <= 1")
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}

    def fit(self, texts: Sequence[str]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for i, text in enumerate(texts):
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, count in tf.items():
                postings[term].append((i, count))
        self.postings = dict(postings)
        n = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        return self

    def scores(self, query: str) -> dict[int, float]:
        """Sparse scores: only documents sharing at least one query term appear.

        Walking postings lists (term-at-a-time) costs O(sum of matching postings)
        rather than O(N) per query, which is the whole point of an inverted index.
        """
        acc: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl or 1.0
        k1, b = self.k1, self.b
        # Repeated query terms count once: "raft raft raft" should not triple-weight.
        for term in dict.fromkeys(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc, tf in plist:
                norm = k1 * (1 - b + b * self.doc_len[doc] / avgdl)
                acc[doc] += idf * tf * (k1 + 1) / (tf + norm)
        return acc

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        # Ties broken by lower index so results are reproducible across runs.
        best = heapq.nsmallest(k, ((-s, d) for d, s in self.scores(query).items()))
        return [(d, -neg) for neg, d in best]
