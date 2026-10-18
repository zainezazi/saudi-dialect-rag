"""Ask a question in Saudi dialect and see the retrieved passages plus a grounded LLM prompt.

    python ask.py "ابي اشحن رصيدي وش الطريقه"
"""
import sys

from saudi_rag import Retriever
from saudi_rag.data import load_docs
from saudi_rag.retriever import build_prompt


def main():
    query = " ".join(sys.argv[1:]) or "النت مره ثقيل ليش"
    hits = Retriever(load_docs()).search(query, k=3)
    print(f"Query: {query}\n")
    for d, s in hits:
        print(f"[{d.id}] {s:.4f}  {d.title}: {d.text}")
    print("\n--- grounded prompt (paste into any chat LLM) ---\n")
    print(build_prompt(query, hits))


if __name__ == "__main__":
    main()
