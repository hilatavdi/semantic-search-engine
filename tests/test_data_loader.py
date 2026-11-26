"""Loading and validating the JSON input file (semantic_search/data_loader.py)."""
import json
from pathlib import Path

import pytest

from semantic_search.data_loader import DataLoadError, load_documents

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "semantic_search_sample_data_en.json"


def write(tmp_path, content, name="data.json"):
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_loads_the_provided_dataset():
    docs = load_documents(SAMPLE)
    assert len(docs) == 60
    assert len({d.id for d in docs}) == 60
    assert docs[0].id == "doc_001" and docs[0].text == "I want to cancel my monthly subscription."


def test_strips_whitespace_and_keeps_optional_category(tmp_path):
    path = write(tmp_path, json.dumps([{"id": "a", "text": "  hello  "}, {"id": "b", "text": "x", "category": "C"}]))
    docs = load_documents(path)
    assert docs[0].text == "hello" and docs[0].category is None
    assert docs[1].category == "C"


def test_accepts_file_saved_with_utf8_bom(tmp_path):
    path = write(tmp_path, b"\xef\xbb\xbf" + json.dumps([{"id": "a", "text": "hi"}]).encode("utf-8"))
    assert len(load_documents(path)) == 1


@pytest.mark.parametrize("content, message", [
    ("{ not json", "Invalid JSON"),
    ('{"id": "a", "text": "hi"}', "list"),
    ("[]", "no entries"),
    ('["just a string"]', "not a JSON object"),
    ('[{"text": "hi"}]', "'id'"),
    ('[{"id": "a", "text": "   "}]', "'text'"),
    ('[{"id": "a", "text": 5}]', "'text'"),
    ('[{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]', "Duplicate id"),
])
def test_invalid_files_raise_clear_error(tmp_path, content, message):
    with pytest.raises(DataLoadError, match=message):
        load_documents(write(tmp_path, content))


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_documents(tmp_path / "does_not_exist.json")
