# Part E - Sentence Pair Similarity

## Setup

The cosine similarity of `all-MiniLM-L6-v2` embeddings, the same score the search engine uses
(`python experiments/sentence_pairs.py`; full output in [`part_e_results.md`](part_e_results.md)).

- The three required pairs share the same first sentence, so only the second sentence changes.
- I added 7 extra pairs to check that the pattern is not a one-off.
- I wrote a predicted score range for every pair **before** running anything. The predictions are in a
  separate, earlier commit than the results.

## Results

| Pair type | Sentence B (A = *"I want to cancel my subscription."*) | Predicted | Cosine |
|---|---|---|---|
| Similar meaning | How do I stop my membership? | 0.60 - 0.80 | **0.649** |
| Same topic, different intent | I want to upgrade my subscription. | 0.50 - 0.70 | **0.692** |
| Unrelated | I am looking for a short hiking trail suitable for children. | 0.00 - 0.15 | **0.019** |

Averages over all 10 pairs: similar meaning **0.648**, same topic / different intent **0.806**,
unrelated **0.021**.

## Did the results match my expectations?

Almost every score fell inside its predicted range. The only exception was *return / buy this
jacket* (0.812, above my range). The **order** of the required pairs, however, was wrong. I expected the paraphrase to score
higher than *upgrade*, but the model rated the opposite request (0.692) as closer than the same
request phrased differently (0.649).

For the extra pairs where only *not* or *on/off* changes, I did predict very high scores, because Part C
had already shown this weakness. What surprised me is that swapping the main verb (*cancel/upgrade*,
*return/buy*) behaves the same way.

In hindsight, my ranges were too wide to be a real test. The prediction that mattered was the
ordering.

## Interpretation

1. The four different-intent pairs keep the same words and structure and change only the action
   (*cancel/upgrade*, *return/buy*, *on/off*) or add a negation (*did not*). They scored 0.69-0.89. The
   paraphrases share almost no words and scored 0.60-0.69. So the model does understand paraphrases,
   but the topic and sentence structure dominate the vector, and the word that flips the meaning moves
   it only a little.
2. Negation is almost invisible: *"I received my refund"* vs. *"I did **not** receive my refund"*
   scored **0.889**, the highest of all pairs.
3. Topics are separated very well. Unrelated pairs scored about 0.02, and every related pair scored at
   least 0.60.
4. The Part D reranker shows the same bias. It scored *cancel / upgrade* +1.45 but *cancel / stop my
   membership* -3.16, which explains why reranking did not fix *upgrade my subscription* in Part D.

## Conclusion

These embeddings tell reliably whether two texts are about the same topic, but they are bad at telling
whether they ask for the same thing. A score threshold would work for filtering out unrelated texts,
but it cannot separate a request from its opposite. This is the cause of the failures in Parts C and
D. Fixing it would need a signal about the relationship between the two sentences, for example
fine-tuning with *cancel / upgrade* style pairs, a contradiction (NLI) model, or an LLM judge.

Since this is only 10 pairs that I chose, on one model, I treat it as a pattern I observed here rather
than a general rule.
