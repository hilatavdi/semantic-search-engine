# Part C - Semantic Search vs. Keyword Search

## Keyword baseline

`semantic_search/keyword_search.py`: lowercase the text, remove stop words (*I*, *my*, *the*, *how*...),
and score each document by the share of query keywords it contains. There is no stemming and no
synonyms.

I ran 7 queries through both engines (`python experiments/compare_search.py`). The full top-5 lists are
in [`part_c_results.md`](part_c_results.md).

| Query | Semantic: relevant in top 5 | Keyword: relevant in top 5 |
|---|---|---|
| my parcel never showed up | 4 / 5 | 0 / 1 |
| I can't log into my account | 3 / 5 | 1 / 5 |
| the app is not working | 3 / 5 | 2 / 4 |
| cancel my booking | 1 / 5 (at #1) | 1 / 3 (3-way tie at 0.5) |
| How do I end my plan? | 2 / 5 (wrong intent at #1) | 1 / 3 (wrong intent at #1) |
| turn off notifications | 0 / 5 | 0 / 2 |
| I want to cancel my subscription | 2 / 5 | 2 / 5 |

"Relevant" means the result would actually help the user (my judgement).

## Where semantic search is better

**"my parcel never showed up"**: none of the relevant documents contain *parcel*, *never* or *showed*.
Keyword search returned a single result, *"Can I pick **up** my order..."*, matched only by the word
*up*. Semantic search returned four shipping-related results in its top 4 (e.g. *"I ordered an item, but it
has not arrived yet"*, 0.519).

**"I can't log into my account"**: keyword search could only match the word *account*, so five
account documents all scored 0.333, and only one of them (*"I forgot my account password"*) was
relevant. It missed *"I cannot **sign in**..."* (a synonym) and
*"...several **login** attempts"* (a different word form). Semantic search ranked exactly these two
first (0.749, 0.708).

Two more cases are in the results file: for *"the app is not working"*, keyword search returned a cafe
"suitable for **working** on a laptop". For *"cancel my booking"* it could not rank three documents
that all contain *cancel*.

## Where semantic search misses the user's intent

**"turn off notifications"**: the top two semantic results are *"I am **not receiving** notifications"*
(0.503) and *"How do I **enable** alerts"* (0.369). Both come from users who want **more**
notifications. The model captured the topic but not the direction of the action. Keyword search did
not find a relevant result either, but its #1 at least shared the action *turn off* (*"How can I turn
off two-factor authentication?"*).

**"I want to cancel my subscription"**: the right answer is #1 (0.941). However, *"How do I
**reactivate** an expired subscription?"* is #3 with 0.684, almost the same score as the relevant #2
(0.688).

## Conclusions

1. Semantic search is much better when users phrase things differently from the documents: synonyms,
   word forms, paraphrases. Keyword search either misses those documents or matches unrelated ones
   because they happen to share a word (like *up* or *working*).
2. Semantic search captures **topic** better than **intent**: opposite requests about the same subject
   score high. This is what Part D tries to improve.
3. Semantic search always returns five results, even when nothing relevant exists, and its scores are
   only a weak signal: 0.50 was an irrelevant top result in one query, while 0.48 was a relevant result
   in another.
