"""Command-line interface for the semantic search engine.

Usage:
    python main.py                          # interactive mode
    python main.py --query "cancel my plan" # one-shot query
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
from semantic_search.search import InvalidQueryError, SemanticSearchEngine, SearchResult

# Resolved relative to this file, so the program works from any current directory.
DEFAULT_DATA = str(Path(__file__).resolve().parent / "data" / "semantic_search_sample_data_en.json")


def print_results(query: str, results: list[SearchResult]) -> None:
    print(f'\nTop {len(results)} results for: "{query}"')
    print("-" * 78)
    for r in results:
        print(f"{r.rank}. [{r.id}]  score={r.score:.4f}  {r.text}")
    print()


def run_query(engine: SemanticSearchEngine, query: str, k: int) -> bool:
    """Run one query and print results. Returns False on a recoverable error."""
    try:
        results = engine.search(query, k)
        print_results(engine.validate_query(query), results)
        return True
    except InvalidQueryError as e:
        print(f"[!] {e}")
    except EmbeddingError as e:
        print(f"[!] Could not embed the query: {e}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic search over a JSON text collection")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to the JSON data file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
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
        print(f"Loading embedding model '{args.model}' ...")
        embedder = SentenceTransformerEmbedder(args.model)
        engine = SemanticSearchEngine(documents, embedder)
    except EmbeddingError as e:
        print(f"[!] {e}", file=sys.stderr)
        return 1
    print(f"Indexed {len(documents)} documents (embedding dim = {engine.doc_vectors.shape[1]})")

    if args.query is not None:
        return 0 if run_query(engine, args.query, args.top_k) else 1

    print("\nType a search query (or 'exit' to quit).")
    while True:
        try:
            query = input("Search> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() in {"exit", "quit", "q"}:
            break
        run_query(engine, query, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
