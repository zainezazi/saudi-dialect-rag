import json
import pathlib

from .retriever import Document

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def load_docs(path=DATA_DIR / "faq.jsonl") -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]


def load_queries(path=DATA_DIR / "eval_queries.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
