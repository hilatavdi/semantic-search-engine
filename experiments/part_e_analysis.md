# Part E - Experimentation & Analysis: Sentence Pair Similarity

## Setup

- **Score:** cosine similarity between `all-MiniLM-L6-v2` embeddings. This is the same model and
  score the search engine uses.
- **The three required pairs** all share the same first sentence, *"I want to cancel my
  subscription."* Only the second sentence changes, so the scores can be compared directly.
- **Extra pairs:** 7 more pairs (2-3 per category) check that the pattern is not a one-off.
- **Predictions first:** I wrote down a predicted cosine range for every pair **before** running anything.
  They are committed in `experiments/sentence_pairs.py` in a separate commit before the results, so the
  Git history shows they were not adjusted afterwards.
- **Comparison only:** the Part D cross-encoder score is shown next to each pair.

Reproduce with `python experiments/sentence_pairs.py`. The full output is in
[`part_e_results.md`](part_e_results.md).

## Results: the three required pairs

| # | Pair type | Sentence B (A = *"I want to cancel my subscription."*) | Predicted | **Cosine** |
|---|---|---|---|---|
| 1 | Similar meaning | How do I stop my membership? | 0.60 - 0.80 | **0.649** |
| 2 | Same topic, different intent | I want to upgrade my subscription. | 0.50 - 0.70 | **0.692** |
| 3 | Unrelated | I am looking for a short hiking trail suitable for children. | 0.00 - 0.15 | **0.019** |

## Results: all pairs, averaged per category

| Category | Pairs | Mean cosine | Range |
|---|---|---|---|
| Similar meaning | 3 | 0.648 | 0.602 - 0.692 |
| **Same topic, different intent** | 4 | **0.806** | 0.692 - 0.889 |
| Unrelated | 3 | 0.021 | 0.019 - 0.023 |

## Do the results match my expectations?

**Each individual score fell inside its predicted range (9 of 10 pairs). However, the most important
expectation, the *order* between categories, was wrong.**

I expected:

> similar meaning **>** same topic / different intent **>** unrelated

The model produced:

> same topic / different intent (0.806) **>** similar meaning (0.648) **>>** unrelated (0.021)

The clearest case is the required pairs themselves:
- *"cancel my subscription"* vs. *"**upgrade** my subscription"*, which means the opposite: **0.692**
- *"cancel my subscription"* vs. *"stop my membership"*, which means the same: **0.649**

**According to the model, the sentence with the opposite request is closer than the one with the same
request.** This is the reverse of what a search engine should do.

My ranges were wide enough that both scores fit inside them. In hindsight, the ranges were a weak test.
The prediction that actually mattered was the **ordering**, and it failed.

## Interpretation

### 1. The model measures shared wording and topic more than intent

Look at what each "different intent" pair has in common:

| Pair | What is shared | What differs | Cosine |
|---|---|---|---|
| I received my refund. / I **did not** receive my refund. | almost every word | one word (*not*) | **0.889** (highest of all pairs) |
| How do I turn **on** notifications? / How do I turn **off** notifications? | every word but one | *on* / *off* | 0.831 |
| I want to **return** this jacket. / I want to **buy** this jacket. | structure + object | the verb | 0.812 |
| I want to **cancel** my subscription. / I want to **upgrade** my subscription. | structure + object | the verb | 0.692 |

Now compare the "similar meaning" pairs, which say the same thing in **different words**:

| Pair | Shared words | Cosine |
|---|---|---|
| I want to cancel my subscription. / How do I stop my membership? | only *my* | 0.649 |
| My package hasn't arrived yet. / I'm still waiting for my delivery. | only *my* | 0.602 |
| I forgot my password. / I can't remember my login details. | only *I*, *my* | 0.692 |

The embedding is clearly not pure word-matching. The paraphrases still score around 0.6-0.7 with
almost no shared words, which is the strength we saw in Part C. However, when two sentences share
structure and topic, a single word that flips the meaning (*not*, *off*, *upgrade*) moves the vector
only slightly.

A sentence embedding compresses the whole sentence into one point. The action word is a small part of
that point, while the topic and the sentence structure make up most of it.

### 2. Negation is almost invisible

*"I received my refund"* vs. *"I did **not** receive my refund"* scored **0.889**, the highest of
all 10 pairs, even though the two sentences describe opposite situations. For a support system, this
is the difference between a happy customer and a complaint.

### 3. Topic separation is excellent

Unrelated pairs scored about **0.02**, and the closest related pair scored 0.602. That is a gap of almost
0.6 with no overlap. The model is very reliable at deciding whether two texts are about the same
subject.

Note that unrelated sentences score about 0, not negative. With this model, about 0 means "nothing in
common", so the useful range of scores is roughly 0 to 1.

### 4. The Part D cross-encoder has the same blind spot

| Pair | Cosine | Cross-encoder |
|---|---|---|
| cancel / stop my membership (same intent) | 0.649 | **-3.16** |
| cancel / upgrade (different intent) | 0.692 | **+1.45** |
| received / did not receive refund | 0.889 | **+6.80** |

The cross-encoder also scores the different-intent pairs **higher** than the paraphrases, and the gap is
even bigger. This explains a result from Part D: reranking did **not** fix *"upgrade my
subscription"*, where *"cancel my subscription"* stayed at #2.

The reranker was trained on MS MARCO to judge whether a passage is *about* the query, not whether it
expresses the *same request*. In Part D it still helped in some cases, probably because
reading both texts together lets it use more context than a single vector. That is a hypothesis I did
not test here. On minimal pairs like these, it shows the
same topic bias as the embedding.

(A caveat: the cross-encoder is not symmetric and was built for query vs. passage, so these scores are
only a rough comparison.)

## Conclusions

1. **Cosine similarity is a good *topic* filter, but a poor *intent* judge.** A threshold could
   reliably separate related from unrelated texts (about 0.02 vs. 0.6 and above). No threshold can
   separate the same intent from a different intent, because different-intent pairs often score
   **higher**.
2. **The failures in Parts C and D are explained by this experiment.** *Reactivate* ranked close to
   *cancel*, and *enable alerts* ranked close to *turn off notifications*. These are not random
   errors; they follow directly from how the embedding behaves on minimal pairs.
3. **An absolute score means little on its own.** A paraphrase scored 0.602, while an opposite statement
   scored 0.889. A high score means "same topic and similar wording", not "same meaning".
4. **What would fix it:** the signal needs to come from something that models the *relationship*
   between the two sentences, not just their topic. Options include:
   - a model trained on contradiction (Natural Language Inference),
   - an intent classifier trained on labelled support requests,
   - fine-tuning the embedding model with "hard negative" pairs like *cancel / upgrade*, so it learns
     to push them apart,
   - an LLM judge that is explicitly asked whether the requested action matches.

## Limitations of this experiment

- **Only 10 pairs**, and I chose them myself. Most different-intent pairs are minimal pairs (one word
  changed), which is the hardest case for embeddings. Different-intent pairs with very different wording
  would probably score lower.
- **One embedding model.** Larger models may handle negation and opposite actions better. This was not
  tested here.
- **Wide prediction ranges** made most individual predictions "correct" by default. In a future
  experiment I would predict the **ranking** of pairs explicitly, because that is what matters for
  search.
