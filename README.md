# Semantic Search Engine

A command-line search engine that finds short texts by **meaning**, not just by matching words. It
also includes a keyword-search baseline for comparison, a cross-encoder reranking step that improves
relevance, and experiments that analyse where embeddings succeed and fail.

- **Embedding model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), run locally
- **Reranker (Part D):** [`cross-encoder/ms-marco-MiniLM-L-12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2), run locally
- **No API keys, no external services.** Everything runs on your machine after a one-time model download.

---

## Contents

1. [Setup](#setup)
2. [Running the search engine](#running-the-search-engine)
3. [Sample inputs and outputs](#sample-inputs-and-outputs)
4. [Embedding model: choice and reasoning](#embedding-model-choice-and-reasoning)
5. [How it works](#how-it-works)
6. [Project structure](#project-structure)
7. [Experiments (Parts C, D, E)](#experiments-parts-c-d-e)
8. [Automated tests](#automated-tests)
9. [Error handling](#error-handling)
10. [Challenges and reflection](#challenges-and-reflection)

---

## Setup

Requires **Python 3.10+**. It was developed and tested with Python 3.12 on Windows 11.

```bash
git clone https://github.com/hilatavdi/semantic-search-engine.git
cd semantic-search-engine

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the models from Hugging Face (about 90 MB for the embedding model and about
130 MB for the reranker). After that they are cached locally, and the program works offline.

> **Windows note:** if creating the virtual environment fails with
> `ModuleNotFoundError: No module named 'pip._internal.utils'`, the project path is probably too long
> for Windows (260-character limit). Create the virtual environment in a shorter path, e.g.
> `python -m venv C:\venvs\semsearch`. See [Challenges](#challenges-and-reflection).

## Running the search engine

```bash
python main.py                                   # interactive semantic search (default)
python main.py --query "my parcel never showed up"   # a single query, then exit
python main.py --mode keyword                    # keyword search only (Part C)
python main.py --mode rerank                     # semantic search + reranking (Part D)
python main.py --mode both                       # semantic and keyword, side by side
python main.py --mode all                        # semantic, keyword and reranked, side by side
python main.py --data other.json --top-k 3       # another data file / number of results
```

In interactive mode, type a query at the `Search>` prompt. Type `exit` to quit.

**Input format:** a JSON array of objects, each with a unique `id` and a `text` (`category` is
optional):

```json
[
  {"id": "doc_001", "text": "I want to cancel my monthly subscription.", "category": "Subscriptions"},
  {"id": "doc_002", "text": "How can I stop recurring charges for this service?", "category": "Subscriptions"}
]
```

The provided dataset (60 customer-support style texts in 8 categories) is in `data/`.

## Sample inputs and outputs

**Semantic search: different words, same meaning.** None of the top results contain *parcel*,
*never* or *showed up*:

```
$ python main.py --query "my parcel never showed up"

[Semantic] Top 5 results for: "my parcel never showed up"
------------------------------------------------------------------------------
1. [doc_021]  score=0.5678  My package was delivered to the wrong address.
2. [doc_017]  score=0.5189  I ordered an item, but it has not arrived yet.
3. [doc_024]  score=0.5121  The tracking status says delivered, but I received nothing.
4. [doc_018]  score=0.4797  Where can I track my package?
5. [doc_025]  score=0.3919  I want to return a product I bought yesterday.
```

**Keyword search on the same query.** It only matches the word *up* (in "pick **up**"):

```
$ python main.py --mode keyword --query "my parcel never showed up"

[Keyword] Top 1 results for: "my parcel never showed up"
------------------------------------------------------------------------------
1. [doc_023]  score=0.2500  Can I pick up my order instead of having it shipped?
```

**Reranking moves the opposite intent ("reactivate") to the bottom:**

```
$ python main.py --mode all --query "I want to cancel my subscription"

[Semantic] Top 5 results for: "I want to cancel my subscription"
------------------------------------------------------------------------------
1. [doc_001]  score=0.9407  I want to cancel my monthly subscription.
2. [doc_005]  score=0.6883  My subscription renewed even though I asked to end it.
3. [doc_007]  score=0.6838  How do I reactivate an expired subscription?
4. [doc_006]  score=0.6279  I want to pause my membership for two months.
5. [doc_052]  score=0.5811  How can I cancel concert tickets I already booked?

[Keyword] Top 5 results for: "I want to cancel my subscription"
------------------------------------------------------------------------------
1. [doc_001]  score=1.0000  I want to cancel my monthly subscription.
2. [doc_003]  score=0.5000  I would like to upgrade to the premium subscription plan.
3. [doc_005]  score=0.5000  My subscription renewed even though I asked to end it.
4. [doc_007]  score=0.5000  How do I reactivate an expired subscription?
5. [doc_008]  score=0.5000  Can I transfer my subscription to another account?

[Reranked] Top 5 results for: "I want to cancel my subscription"
------------------------------------------------------------------------------
1. [doc_001]  score=8.0499  I want to cancel my monthly subscription.
2. [doc_006]  score=1.1436  I want to pause my membership for two months.
3. [doc_005]  score=-0.6153  My subscription renewed even though I asked to end it.
4. [doc_012]  score=-1.4406  I want to permanently delete my user account.
5. [doc_007]  score=-4.5541  How do I reactivate an expired subscription?
```

The scores use different scales in each mode:
- **Semantic:** cosine similarity, roughly 0 to 1.
- **Keyword:** the share of query keywords found in the text, from 0 to 1.
- **Reranked:** a raw cross-encoder logit. Only its order matters, not its absolute value.

**Error handling:**

```
$ python main.py --query "   "
[!] Query is empty - please type something to search for

$ python main.py --data broken.json
[!] Failed to load data: Invalid JSON in broken.json (line 1, col 3): Expecting property name enclosed in double quotes
```

## Embedding model: choice and reasoning

I chose **`all-MiniLM-L6-v2`** (sentence-transformers) for these reasons:

| Reason | Detail |
|---|---|
| **Fits the data** | It is trained specifically for *sentence-level* similarity on over 1 billion sentence pairs. The dataset consists of short, single-sentence requests, so it matches what the model was built for. |
| **Runs locally** | No API key and no cost per query. The data never leaves the machine, and results are fully reproducible. Reviewers can run the project without credentials. |
| **Small and fast** | About 90 MB with 384-dimensional vectors. It embeds the whole dataset in about 0.25 s on a laptop CPU, and a query takes about 20 ms. |
| **Well known baseline** | It is one of the most widely used open embedding models, so its behaviour is well documented and easy to compare against. |

**Trade-offs:**
- It is less accurate than large hosted models, such as OpenAI's `text-embedding-3` or larger open
  models like `bge-large` / `e5-large`.
- It is mainly trained on English.
- It truncates inputs longer than about 256 word-pieces, which is irrelevant for short texts like these.

For a 60-document assignment, I preferred a model that anyone can run, test and reproduce over a few
extra points of accuracy. The embedder sits behind a small interface (`--model` flag, `Embedder`
protocol), so it can be swapped without changing the search logic.

## How it works

```
                      startup (once)                               per query
data/*.json ──> load & validate ──> embed all 60 texts ──> [60 x 384 matrix in memory]
                                                                    │
"my parcel never showed up" ──> validate ──> embed query ──> cosine similarity vs. all 60
                                                                    │
                                                         sort ──> top 5 (id, text, score)
                                    (--mode rerank: top 15 ──> cross-encoder ──> top 5)
```

1. **Text embeddings:** the model maps each text to a 384-number vector. Texts with similar meaning get
   vectors that point in similar directions.
2. **Cosine similarity:** this measures the angle between the query vector and each document vector:
   1 means the same direction, about 0 means unrelated. Vectors are normalised once, so cosine becomes
   a fast dot product.
3. **Ranking:** the documents are sorted by similarity and the top *k* are returned. Ties keep
   dataset order, so results are deterministic.
4. **Storage:** embeddings are kept in memory. For 60 documents, re-computing them at startup takes
   well under a second, which is simpler than managing a cache file or a vector database.

## Project structure

```
main.py                          CLI: argument parsing, interactive loop, error messages
semantic_search/
    data_loader.py               load + validate the JSON file           (data loading)
    embeddings.py                embedding model wrapper                 (embedding generation)
    ranking.py                   cosine similarity, top-k (pure numpy)   (ranking)
    search.py                    SemanticSearchEngine, query validation  (searching)
    keyword_search.py            keyword baseline                        (Part C)
    reranker.py                  cross-encoder reranking                 (Part D)
experiments/
    compare_search.py            Part C: semantic vs keyword  -> part_c_results.md, part_c_analysis.md
    rerank_before_after.py       Part D: before/after rerank  -> part_d_results.md, part_d_analysis.md
    sentence_pairs.py            Part E: sentence pair scores -> part_e_results.md, part_e_analysis.md
tests/                           42 automated tests (pytest), fake models, no network
data/semantic_search_sample_data_en.json
```

The ranking math (`ranking.py`, `rerank()`) never imports a model. The search engines receive the
model as a parameter (dependency injection), so tests can pass in a fake one.

## Experiments (Parts C, D, E)

Each experiment is a script that regenerates its results file. The analysis is in a separate document.

| Part | Question | Key finding | Details |
|---|---|---|---|
| **C** | Semantic vs. keyword search on 7 queries | Semantic search wins on synonyms and paraphrases (4 cases). It fails on **intent**: for *"turn off notifications"* it returns results about *enabling* notifications. | [analysis](experiments/part_c_analysis.md) · [results](experiments/part_c_results.md) |
| **D** | Can a cross-encoder reranker fix intent errors? | **Partly.** It helped in 3 of 8 queries (e.g. the right answer moved from #5 to #1 for *"stop getting emails from you"*), made no change in 4, and was slightly worse in 1. It did **not** fix *upgrade* vs. *cancel*. It is about 10x slower. | [analysis](experiments/part_d_analysis.md) · [results](experiments/part_d_results.md) |
| **E** | How similar are paraphrases, same-topic/different-intent pairs, and unrelated pairs? | **Opposite-intent pairs score *higher* than paraphrases** (mean 0.81 vs. 0.65). *"I received my refund"* vs. *"I did **not** receive my refund"* scored 0.89. Embeddings separate topics well, but not intent. | [analysis](experiments/part_e_analysis.md) · [results](experiments/part_e_results.md) |

```bash
python experiments/compare_search.py
python experiments/rerank_before_after.py
python experiments/sentence_pairs.py
```

## Automated tests

```bash
python -m pytest            # 42 tests, under 1 second, no model download needed
```

The tests replace the real models with simple fakes (`tests/fakes.py`). A *bag-of-words embedder*
gives predictable similarities, a *failing embedder* simulates an embedding error, and a *fixed-score
reranker* replaces the cross-encoder. They cover:

| File | What is tested |
|---|---|
| `test_ranking.py` | cosine values (same / orthogonal / opposite), length invariance, zero vectors, top-k order, deterministic ties |
| `test_data_loader.py` | the real dataset loads; a missing file and 8 kinds of invalid files are rejected with clear messages; BOM files are accepted |
| `test_search.py` | ranking by similarity; top 5 by default; documents embedded once; empty/invisible/too-long queries rejected; embedding failures (exception, NaN output, wrong shape) reported; keyword scoring; reranking order, ties and errors |

To check that the tests can actually catch bugs, I planted two bugs on purpose: reversing the sort
order, and removing the empty-query check. The tests failed (3 and 6 failures), and they passed again
once the code was restored.

## Error handling

| Situation | Behaviour |
|---|---|
| File missing, invalid JSON, not a list, empty list | clear `[!] Failed to load data: ...` message, exit code 1 |
| Entry without a non-empty string `id` / `text`, or duplicate `id` | message naming the entry, exit code 1 |
| File saved with a UTF-8 BOM (produced by some Windows tools) | accepted |
| Empty query, whitespace only, or invisible characters only | `[!] Query is empty` message; interactive mode keeps running |
| Query over 1000 characters | rejected with a message |
| Model cannot be downloaded / wrong model name | `[!] Could not load embedding model ...` with a hint (internet, model name), no traceback |
| Model fails or returns NaN / the wrong number of vectors | `EmbeddingError` is caught and reported, without a crash |
| Reranker fails to load or score | `RerankerError` is caught and reported |
| Query has only stop words (keyword mode) | `(no matching documents)` |

## Challenges and reflection

### Challenge 1: invisible characters, in the data and in the code

**Problem:** while testing error handling, a query piped from PowerShell arrived with an invisible
byte-order mark (BOM, `U+FEFF`) at the start. Python's `str.strip()` does not remove it, so a query that
*looked* empty passed validation and was searched as a real query. The same issue appeared in input
files: a JSON file saved with a BOM (some Windows tools do this, e.g. PowerShell's `Out-File -Encoding utf8`)
was rejected as "invalid JSON".

**How I fixed it:**
- The loader now reads files with `utf-8-sig`, which accepts an optional BOM.
- Query validation removes a list of invisible characters before checking whether the query is empty.

Later, while reviewing the code in VS Code, I noticed that the list of invisible characters itself had
been written into the source file *as literal invisible characters*. It worked, but it was unreadable and
could be silently broken by an editor. I rewrote it as explicit code points
(`[0xFEFF, 0x200B, ...]`) and scanned every file in the repository for hidden characters.

**Lesson:** text that "looks empty" is not always empty, and code should be readable as well as correct.
The BOM file and invisible-character query cases are now covered by automated tests.

### Challenge 2: Windows path length broke the environment setup

**Problem:** creating the virtual environment inside the project folder failed with
`ModuleNotFoundError: pip._internal.utils`. The project lived in a deep OneDrive path. My diagnosis:
combined with pip's own nested folders, file paths went over the 260-character Windows limit, so pip
was installed incomplete. Creating the same environment in a short path worked immediately, which
supports this explanation.

**How I fixed it:** I created the virtual environment in a short path outside the project, and
resolved the data file path relative to `main.py`, so the program runs from any folder. I documented
this in [Setup](#setup).

### Challenge 3: the improvement in Part D did not fully work

**Problem:** I expected the cross-encoder to fix the *cancel vs. upgrade* confusion from the
assignment brief. It fixed *cancel vs. reactivate*, but not *upgrade vs. cancel*.

**How I addressed it:**
- Instead of only showing the cases that worked, I evaluated 8 queries with relevance labels decided
  in advance, and reported improvements, no-changes and a regression.
- Part E then explained *why*. On minimal pairs, both models score opposite intents higher than
  paraphrases, because they respond to shared topic and wording more than to the action.

**Lesson:** an improvement should be measured, not assumed. A partial result that is explained is more
useful than a demo that only shows the wins.

### What I would improve with more time

1. **A proper evaluation set:** 50-100 queries with labelled relevant documents, and standard metrics
   (MRR, nDCG@5) instead of my small, hand-judged set.
2. **Better intent handling:**
   - fine-tune the embedding model or the cross-encoder with "hard negative" pairs (*cancel* vs.
     *upgrade*), or
   - add an NLI / intent-classification step, or
   - try an LLM judge on the top 5 and compare its quality, cost and latency.
3. **A "no good match" answer:** semantic search always returns 5 results, even when nothing is
   relevant (e.g. *"turn off notifications"*). A calibrated score threshold would allow an honest
   "nothing found".
4. **Hybrid + rerank:** keyword search kept action words like *turn off* that embeddings
   under-weight. Combining the keyword and semantic candidate lists before reranking could improve
   recall.
5. **Scaling:** cache embeddings to disk, and use a vector index (e.g. FAISS) for large collections.
