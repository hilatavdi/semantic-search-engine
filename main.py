"""Command-line interface for the search engine.

Usage:
    python main.py                               # interactive semantic search
    python main.py --mode keyword                # interactive keyword search
    python main.py --mode rerank                 # semantic search + cross-encoder reranking (Part D)
    python main.py --mode both                   # semantic and keyword results side by side
    python main.py --mode all                    # semantic, keyword and reranked results
    python main.py --query "cancel my plan"      # one-shot query
    python main.py --data other.json --top-k 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quiet third-party noise (HF Hub symlink/token warnings, weight-loading progress bars).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from semantic_search.data_loader import DataLoadError, load_documents
from semantic_search.embeddings import DEFAULT_MODEL, EmbeddingError, SentenceTransformerEmbedder
from semantic_search.keyword_search import KeywordSearchEngine
from semantic_search.reranker import (DEFAULT_RERANKER, CrossEncoderReranker, RerankerError,
                                      RerankingSearchEngine)
from semantic_search.search import InvalidQueryError, SemanticSearchEngine, SearchResult

# Resolved relative to this file, so the program works from any current directory.
DEFAULT_DATA = str(Path(__file__).resolve().parent / "data" / "semantic_search_sample_data_en.json")


def print_results(label: str, query: str, results: list[SearchResult]) -> None:
    print(f'\n[{label}] Top {len(results)} results for: "{query}"')
    print("-" * 78)
    if not results:
        print("(no matching documents)")
    for r in results:
        print(f"{r.rank}. [{r.id}]  score={r.score:.4f}  {r.text}")
    print()


def run_query(engines: dict, query: str, k: int) -> bool:
    """Run one query through every engine and print results. Returns False on a recoverable error."""
    try:
        clean = SemanticSearchEngine.validate_query(query)
        for label, engine in engines.items():
            print_results(label, clean, engine.search(clean, k))
        return True
    except InvalidQueryError as e:
        print(f"[!] {e}")
    except EmbeddingError as e:
        print(f"[!] Could not embed the query: {e}")
    except RerankerError as e:
        print(f"[!] Could not rerank the results: {e}")
    return False


def build_engines(mode: str, documents, model_name: str, reranker_name: str) -> dict:
    engines = {}
    semantic = None
    if mode in ("semantic", "rerank", "both", "all"):
        print(f"Loading embedding model '{model_name}' ...")
        semantic = SemanticSearchEngine(documents, SentenceTransformerEmbedder(model_name))
        print(f"Indexed {len(documents)} documents (embedding dim = {semantic.doc_vectors.shape[1]})")
        if mode != "rerank":
            engines["Semantic"] = semantic
    if mode in ("keyword", "both", "all"):
        engines["Keyword"] = KeywordSearchEngine(documents)
    if mode in ("rerank", "all"):
        print(f"Loading reranker model '{reranker_name}' ...")
        engines["Reranked"] = RerankingSearchEngine(semantic, CrossEncoderReranker(reranker_name))
    return engines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic / keyword search over a JSON text collection")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to the JSON data file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER, help="cross-encoder model for --mode rerank/all")
    parser.add_argument("--mode", choices=["semantic", "keyword", "rerank", "both", "all"], default="semantic",
                        help="semantic | keyword | rerank (semantic + reranking) | both (semantic + keyword) "
                             "| all (all three). Default: semantic")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show (default 5)")
    parser.add_argument("--query", help="Run a single query and exit")
    args = parser.parse_args(argv)

    if args.top_k < 1:
        print("[!] --top-k must be at least 1", file=sys.stderr)
        return 2

    try:
        documents = load_documents(args.data)
    except DataLoadError as e:
        print(f"[!] Failed to load data: {e}", file=sys.stderr)
        return 1
    print(f"Loaded {len(documents)} documents from {args.data}")

    try:
        engines = build_engines(args.mode, documents, args.model, args.reranker)
    except (EmbeddingError, RerankerError) as e:
        print(f"[!] {e}", file=sys.stderr)
        return 1

    if args.query is not None:
        return 0 if run_query(engines, args.query, args.top_k) else 1

    print("\nType a search query (or 'exit' to quit).")
    while True:
        try:
            query = input("Search> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() in {"exit", "quit", "q"}:
            break
        run_query(engines, query, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
