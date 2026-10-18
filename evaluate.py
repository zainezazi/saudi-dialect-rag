"""Compare retrieval modes on the dialect query set: Recall@1, Recall@3, MRR."""
import argparse

from saudi_rag import Retriever
from saudi_rag.data import DATA_DIR, load_docs, load_queries


def evaluate(mode: str, docs, queries, k: int = 3):
    r = Retriever(docs, mode=mode)
    r1 = r3 = mrr = 0.0
    misses = []
    for q in queries:
        ids = [d.id for d, _ in r.search(q["query"], k=len(docs))]
        rank = ids.index(q["gold"]) + 1
        r1 += rank == 1
        r3 += rank <= k
        mrr += 1 / rank
        if rank > 1:
            misses.append((q["query"], q["gold"], ids[0], rank))
    n = len(queries)
    return {"mode": mode, "recall@1": r1 / n, f"recall@{k}": r3 / n, "mrr": mrr / n}, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--heldout", action="store_true",
                    help="use queries written after the dialect lexicon was frozen")
    args = ap.parse_args()
    docs = load_docs()
    if args.heldout:
        queries = load_queries(DATA_DIR / "heldout_queries.jsonl")
        label = "held-out"
    else:
        queries = load_queries()
        label = "dev"
    print(f"{len(docs)} passages, {len(queries)} Saudi-dialect {label} queries\n")
    print(f"{'mode':<22}{'Recall@1':>10}{'Recall@3':>10}{'MRR':>8}")
    for mode in Retriever.MODES:
        m, misses = evaluate(mode, docs, queries)
        print(f"{mode:<22}{m['recall@1']:>10.2f}{m['recall@3']:>10.2f}{m['mrr']:>8.2f}")
        if args.show_misses:
            for q, gold, top, rank in misses:
                print(f"    miss: {q!r} gold={gold} top={top} gold_rank={rank}")


if __name__ == "__main__":
    main()
