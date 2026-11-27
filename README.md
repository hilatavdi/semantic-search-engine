# Semantic Search Engine

A command-line search engine that finds short texts by meaning, compared against a keyword-search
baseline, with a cross-encoder reranking step and a few experiments on where embeddings succeed and fail.

Everything runs locally: no API keys, no external services after a one-time model download.

## Where to find each part

| Part | Where |
|---|---|
| **A** - Independent learning | [`docs/part_a_learning.md`](docs/part_a_learning.md) |
| **B** - Semantic search engine | `main.py`, `semantic_search/` (this README) |
| **C** - Semantic vs. keyword search | [`experiments/part_c_analysis.md`](experiments/part_c_analysis.md) |
| **D** - Improving search quality | [`experiments/part_d_analysis.md`](experiments/part_d_analysis.md) |
| **E** - Sentence pair experiment | [`experiments/part_e_analysis.md`](experiments/part_e_analysis.md) |
| **F** - Tests, README, reflection | `tests/`, this README |

The `experiments/*_results.md` files are the raw output of the experiment scripts.

## Setup

Tested with Python 3.12 on Windows 11.

```bash
git clone https://github.com/hilatavdi/semantic-search-engine.git
cd semantic-search-engine
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The first run downloads the models (about 90 MB + 130 MB) from Hugging Face. After that they are
cached locally.

## Running

```bash
python main.py                                       # interactive: type a query, "exit" to quit
python main.py --query "my parcel never showed up"   # one query
python main.py --mode keyword                        # keyword search (Part C)
python main.py --mode rerank                         # semantic + reranking (Part D)
python main.py --mode both                           # semantic and keyword side by side
python main.py --mode all                            # all three side by side
python main.py --data other.json --top-k 3
```

The input is a JSON list of objects with a unique `id` and a `text`. The provided dataset (60 texts)
is in `data/`.

## Sample output

Real program output (startup lines omitted). None of the semantic results contain *parcel*, *never*
or *showed up*. Keyword search only matches the word *up*:

```
$ python main.py --mode both --query "my parcel never showed up"

[Semantic] Top 5 results for: "my parcel never showed up"
------------------------------------------------------------------------------
1. [doc_021]  score=0.5678  My package was delivered to the wrong address.
2. [doc_017]  score=0.5189  I ordered an item, but it has not arrived yet.
3. [doc_024]  score=0.5121  The tracking status says delivered, but I received nothing.
4. [doc_018]  score=0.4797  Where can I track my package?
5. [doc_025]  score=0.3919  I want to return a product I bought yesterday.

[Keyword] Top 1 results for: "my parcel never showed up"
------------------------------------------------------------------------------
1. [doc_023]  score=0.2500  Can I pick up my order instead of having it shipped?
```

Errors are reported without a crash:

```
$ python main.py --query "   "
[!] Query is empty - please type something to search for

$ python main.py --data broken.json
[!] Failed to load data: Invalid JSON in broken.json (line 1, col 3): Expecting property name enclosed in double quotes
```

The program also handles a missing file, invalid or duplicate ids, empty texts, and a model that
cannot be loaded or returns invalid output.

## Embedding model

I chose `sentence-transformers/all-MiniLM-L6-v2`. It is trained for sentence-level similarity, which
fits short, single-sentence requests like these. It runs locally, so there is no API key or cost, and
anyone can reproduce the results. It is also small and fast: about 90 MB, 384-dimensional vectors, and
about 20 ms per query on a laptop CPU.

The trade-off is lower accuracy than large hosted models, and it is mainly trained on English. For 60
short texts I preferred simplicity and reproducibility.

## Code structure

```
semantic_search/data_loader.py      load + validate the JSON file
semantic_search/embeddings.py       embedding model wrapper
semantic_search/ranking.py          cosine similarity + top-k (no model, easy to test)
semantic_search/search.py           semantic search engine, query validation
semantic_search/keyword_search.py   keyword baseline (Part C)
semantic_search/reranker.py         cross-encoder reranking (Part D)
```

The documents are embedded once at startup and kept in memory.

## Tests

```bash
python -m pytest     # 45 tests, under a second, no model download
```

The tests replace the real models with small fakes, so the ranking logic is tested without the
network. They cover the ranking math, invalid input files, empty and invisible-character queries,
embedding failures, keyword scoring, and reranking.

## Challenges and reflection

**Invisible characters.** A query piped from PowerShell started with an invisible byte-order mark
(BOM). `strip()` does not remove it, so a query that looked empty was searched. JSON files saved with
a BOM were also rejected. I now strip invisible characters from queries and read files with
`utf-8-sig`. Both cases have tests.

**Windows path length.** Creating the virtual environment inside the deep OneDrive project folder
failed (`ModuleNotFoundError: pip._internal.utils`). Most likely the paths went over the
260-character limit. Creating it in a short path fixed it.

**The improvement only partly worked.** I expected reranking to fix the *cancel* vs. *upgrade*
confusion. It helped in some cases but not that one. I tested it on 8 queries and wrote down the cases
where it didn't help or got slightly worse, too. The sentence-pair experiment (Part E) then explained
why: both models react to shared topic and wording more than to the requested action.

**With more time I would:**
- build a larger labelled query set and use standard ranking metrics such as MRR
- try fine-tuning with *cancel* / *upgrade* style pairs, or an LLM judge on the top results
- return "no good match" when even the best score is low
