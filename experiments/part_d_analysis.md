# Part D - Improving Search Quality: Cross-Encoder Reranking

## The problem

Part C showed that semantic search captures the topic of a query better than its intent. For example,
*"reactivate an expired subscription"* scored almost as high as a relevant result for *"I want to
cancel my subscription"*.

## The approach

Two-stage search (`semantic_search/reranker.py`, `python main.py --mode rerank`):
1. Semantic search returns the top **15** candidates (unchanged).
2. A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-12-v2`) scores each (query, candidate) pair,
   and the top 5 by that score are returned.

**Why a cross-encoder.** The embedding model (a *bi-encoder*) turns each text into a vector
separately, before any comparison. The shared topic dominates that vector, and one word like *cancel*
vs. *reactivate* changes it only slightly. A cross-encoder reads the query and the candidate
**together**, so it can compare the two actions directly. It is slower (one model run per pair), so it
only scores the top 15. This retrieve-then-rerank pattern is the standard way to use it.

**Advantages:**
- It targets the exact weakness found in Part C.
- It keeps the strengths of semantic search, because stage 1 is unchanged.
- It is a small local model: no API key, no cost, deterministic.

**Alternatives I tried first:**
- **Hybrid** (0.7 x semantic + 0.3 x keyword): almost no effect on intent, and it brought back keyword
  mistakes.
- **A cross-encoder trained on duplicate questions** (Quora): poor rankings.
- **The smaller L-6 version of the chosen model:** similar, but worse on the login query.

An LLM judge would probably handle intent best, but it needs an API key, costs money per query, and is
harder to reproduce, so I did not try it.

## Results: before vs. after

8 queries, with the relevant documents for each written in the script
(`python experiments/rerank_before_after.py`; full tables in [`part_d_results.md`](part_d_results.md)).
I wrote these labels after some exploratory runs, so this is a rough check, not a blind test.

| Query | First relevant result (before -> after) | Effect |
|---|---|---|
| stop getting emails from you | #5 -> **#1** | clearly better |
| How do I end my plan? | #3 -> **#1** | better |
| I want to cancel my subscription | #1 -> #1 | better order below #1 (see example) |
| I can't log into my account | #1 -> #1 | slightly worse: a relevant result dropped from #3 to #5 |
| I want to upgrade my subscription | #1 -> #1 | not fixed: *cancel* is still #2 |
| my parcel never showed up / cancel my booking (2 queries) | #1 -> #1 | no change |
| turn off notifications | not in top 5 -> not in top 5 | no fix: the only partly relevant document (*"stop receiving course recommendations"*) stays out of the top 5 |

**Example: "I want to cancel my subscription"**

| # | Before (cosine) | After (cross-encoder score) |
|---|---|---|
| 1 | cancel my monthly subscription (0.941) | cancel my monthly subscription (8.05) |
| 2 | renewed even though I asked to end it (0.688) | pause my membership (1.14) |
| 3 | **reactivate** an expired subscription (0.684) | renewed even though I asked to end it (-0.62) |
| 4 | pause my membership (0.628) | permanently delete my user account (-1.44) |
| 5 | cancel concert tickets (0.581) | **reactivate** an expired subscription (-4.55) |

The opposite intent (*reactivate*) dropped from #3 to #5 and got the lowest score. The unrelated
concert tickets left the top 5.

**Example: "stop getting emails from you"**. Before, the #1 result was *"How do I change the **email**
address..."*, a topic match, and the right answer, *"I want to **stop receiving** course
recommendations"*, was only #5. After reranking, it moved to #1. It is the only document with the
same intent.

## Did it improve the results?

**Partly.** It helped in 3 of 8 queries, all of them the kind of intent problem found in Part C.
4 queries stayed about the same, and 1 got slightly worse. It did **not** fix *upgrade* vs. *cancel*.
The reranker was trained on MS MARCO (web questions and the passages that answer them), so it judges whether a text
is *about* the query, not whether it is the *same request*. Part E confirms this.

## Limitations

- It is about 10x slower: about 21 ms per query vs. about 218 ms with reranking, on a laptop CPU.
- It can only re-order the 15 candidates. If the right document is not among them, reranking cannot
  find it.
- The scores are raw logits. They show the order within one query, not an absolute
  "relevant / not relevant" threshold.
- The evaluation is small and hand-labelled (only 8 queries).
