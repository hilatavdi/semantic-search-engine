"""Loading and validating the JSON document collection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class DataLoadError(Exception):
    """Raised when the input file is missing, malformed, or has invalid entries."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    category: str | None = None


def load_documents(path: str | Path) -> list[Document]:
    """Read a JSON array of {"id", "text", ["category"]} objects.

    Raises DataLoadError with a human-readable message on any problem.
    """
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"Input file not found: {path}")

    try:
        # utf-8-sig also accepts files saved with a UTF-8 BOM (some Windows tools add one).
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in {path.name} (line {e.lineno}, col {e.colno}): {e.msg}") from e
    except UnicodeDecodeError as e:
        raise DataLoadError(f"{path.name} is not valid UTF-8 text") from e

    if not isinstance(raw, list):
        raise DataLoadError("Expected the JSON root to be a list of entries")
    if not raw:
        raise DataLoadError("The input file contains no entries")

    documents: list[Document] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise DataLoadError(f"Entry #{i} is not a JSON object")

        doc_id = entry.get("id")
        text = entry.get("text")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise DataLoadError(f"Entry #{i} is missing a non-empty string 'id'")
        if not isinstance(text, str) or not text.strip():
            raise DataLoadError(f"Entry '{doc_id}' is missing a non-empty string 'text'")
        if doc_id in seen_ids:
            raise DataLoadError(f"Duplicate id '{doc_id}' (ids must be unique)")
        seen_ids.add(doc_id)

        category = entry.get("category")
        documents.append(Document(id=doc_id, text=text.strip(),
                                  category=category if isinstance(category, str) else None))

    return documents
