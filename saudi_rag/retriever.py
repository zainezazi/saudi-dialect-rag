"""Hybrid retriever: word-level BM25 + character n-gram BM25, fused with RRF.

Word BM25 rewards exact (normalized, stemmed) matches. Character n-grams
catch broken plurals and spelling drift that a light stemmer misses
("الفرع" vs "الفروع"). Reciprocal Rank Fusion combines the two rankings
without having to calibrate their scores.
"""
from dataclasses import dataclass

from .bm25 import BM25
from .dialect import expand_query
from .normalize import normalize, tokenize


@dataclass
class Document:
    id: str
    title: str
    text: str


def char_ngrams(text: str, n: int = 3) -> list[str]:
    grams = []
    for word in normalize(text).split():
        w = f"#{word}#"
        grams += [w[i:i + n] for i in range(max(len(w) - n + 1, 1))]
    return grams


class Retriever:
    MODES = ("raw", "normalized", "normalized+dialect", "hybrid")

    def __init__(self, docs: list[Document], mode: str = "hybrid", rrf_k: int = 60):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.docs, self.mode, self.rrf_k = docs, mode, rrf_k
        full = [f"{d.title} {d.text}" for d in docs]
        if mode == "raw":
            self.word = BM25([t.split() for t in full])
        else:
            self.word = BM25([tokenize(t) for t in full])
        if mode == "hybrid":
            self.chars = BM25([char_ngrams(t) for t in full])

    def _rank(self, scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        if self.mode == "raw":
            scores = self.word.scores(query.split())
            order = self._rank(scores)
            return [(self.docs[i], scores[i]) for i in order[:k]]
        q = expand_query(query) if self.mode != "normalized" else query
        word_scores = self.word.scores(tokenize(q))
        if self.mode != "hybrid":
            order = self._rank(word_scores)
            return [(self.docs[i], word_scores[i]) for i in order[:k]]
        char_scores = self.chars.scores(char_ngrams(q))
        fused = [0.0] * len(self.docs)
        for scores in (word_scores, char_scores):
            for rank, i in enumerate(self._rank(scores)):
                fused[i] += 1.0 / (self.rrf_k + rank + 1)
        order = self._rank(fused)
        return [(self.docs[i], fused[i]) for i in order[:k]]


def build_prompt(query: str, hits: list[tuple[Document, float]]) -> str:
    """Grounded prompt for any chat LLM: answer only from the cited passages."""
    context = "\n".join(f"[{d.id}] {d.title}: {d.text}" for d, _ in hits)
    return (
        "أنت مساعد خدمة عملاء. أجب باللهجة السعودية وباختصار، "
        "واعتمد فقط على المقاطع التالية واذكر رقم المقطع. "
        "إذا لم تجد الإجابة فقل إنك لا تعرف.\n\n"
        f"المقاطع:\n{context}\n\nسؤال العميل: {query}\nالإجابة:"
    )
