# Part A - Independent Learning

## The four concepts, and where they appear in my code

### 1. Semantic search

Semantic search finds documents by **meaning**:
1. Embed all the documents (once) - see *Text embeddings* below.
2. Embed the user's query with the same model.
3. Compute the similarity between the query and every document - see *Cosine similarity* below.
4. Return the documents with the highest similarity.

**In my project:** this is exactly `SemanticSearchEngine.search()` in `semantic_search/search.py`.
For example, for *"my parcel never showed up"* it returns *"I ordered an item, but it has not arrived
yet"*, even though the two sentences share no important words.

### 2. Text embeddings

An embedding model turns a piece of text into a list of numbers, a **vector**. The model is trained so
that texts with similar meaning get vectors that are close to each other, and texts about unrelated
things get vectors that are far apart. You can think of every sentence as a point in a space with many
dimensions, where "close" means "similar meaning".

**In my project:** I use `all-MiniLM-L6-v2`, which turns every sentence into **384 numbers**
(`semantic_search/embeddings.py`). When the program starts, it embeds all 60 texts once and keeps the
60 x 384 matrix in memory (`semantic_search/search.py`). Every query is embedded with the same model,
so the query and the documents are points in the same space and can be compared.

One thing I learned from the experiments (Part E): the vector captures the **topic** of a sentence very
well, but small words that flip the meaning (*not*, *off*, *cancel* vs. *upgrade*) change it only a
little.

### 3. Cosine similarity

Cosine similarity measures the **angle** between two vectors, not their length:

```
cos(A, B) = (A . B) / (|A| x |B|)
```

- 1 = the vectors point in the same direction (very similar meaning)
- 0 = unrelated
- -1 = opposite directions (in practice, with this model, unrelated sentences score about 0, not negative)

The angle is used rather than the distance because the direction of the vector carries the meaning,
while its length mostly does not.

**In my project:** `semantic_search/ranking.py` normalises every vector to length 1. After that, the
cosine is just a dot product, so comparing a query to all 60 documents is a single matrix
multiplication. The scores are then sorted and the top 5 are returned (`top_k`). The automated tests
check the basic cases: the same direction gives 1, perpendicular gives 0, opposite gives -1, and the
length of the vector does not matter.

### 4. Semantic search vs. keyword search

| | Keyword search | Semantic search |
|---|---|---|
| Matches | the same **words** | the same **meaning** |
| Synonyms (*parcel* / *package*) | missed | found |
| Different word forms (*login* / *log in*) | missed (without stemming) | found |
| Ranking when several documents match | weak: many get the same score | graded scores |
| "No results" | yes, when no word matches | never: it always returns the closest texts |
| Opposite intent on the same topic | also confused | confused (*cancel* close to *reactivate*) |
| Explainable? | yes, you can see which words matched | harder: the score comes from the model |

**In my project:** I implemented a basic keyword search (`semantic_search/keyword_search.py`) and ran
the same queries through both (Part C). Semantic search was clearly better when users phrase things
differently from the documents. Keyword search had one real advantage: it keeps exact action words like
*"turn off"*, which the embedding gives little weight to.

### How the concepts connect

```
text --(embedding model)--> vector --(cosine similarity with every document)--> scores --(sort)--> top 5
```

Embeddings turn meaning into numbers, cosine similarity compares those numbers, and semantic search is
the whole pipeline built from the two. Keyword search skips the embedding step and compares the words
directly. That is why it is more predictable, but it does not understand meaning.

## Resources I used

- **"What are text embeddings?"**: a video by Google Cloud Tech (YouTube). It gave me the intuition
  for embeddings as points in a space, where texts with similar meaning are close to each other.
- **Sentence-Transformers documentation** (sbert.net): the "Semantic Search" and "Cross-Encoders"
  pages. I used them to understand how to embed text, compute similarity, the difference between
  bi-encoders and cross-encoders, and how retrieve-then-rerank works.
- **Hugging Face model card for `all-MiniLM-L6-v2`**: model size, vector dimension (384), maximum input
  length, and what data it was trained on.
- **Hugging Face model card for `cross-encoder/ms-marco-MiniLM-L-12-v2`**: what the reranker was trained
  on (MS MARCO) and how to use it.
- **Wikipedia, "Cosine similarity"**: the formula and why it ignores vector length.
- **AI assistants:**
  - **Gemini**: helped me understand the assignment and the concepts, and gave general guidance along
    the way.
  - **Claude Code**: helped write and review the code, run the experiments, and check the analyses. I reviewed and tested the results myself.
