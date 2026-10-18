"""saudi_rag: a tiny, dependency-free retrieval pipeline for Saudi-dialect Arabic questions."""
from .normalize import normalize, tokenize
from .dialect import expand_query
from .retriever import Retriever, Document

__all__ = ["normalize", "tokenize", "expand_query", "Retriever", "Document"]
