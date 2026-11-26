# Part C – Semantic Search vs. Keyword Search

## The keyword baseline

`semantic_search/keyword_search.py` implements a deliberately basic keyword search:

1. Lowercase the text and split it into words.
2. Remove stop words (`I`, `my`, `the`, `how`, `want`, ...).
3. **Score = share of the query's keywords that appear in the document** (1.0 = all of them, 0.5 = half).
   Documents with no matching keyword are not returned at all.

There is no stemming (`login` ≠ `log`), no synonyms (`parcel` ≠ `package`) and no weighting of rare words.

## Experiment

The same 7 queries were run through both engines with
`python experiments/compare_search.py`. The full top-5 tables, including the category of every
result, are in [`part_c_results.md`](part_c_results.md). The comparisons below are excerpts from them.

| # | Query | Semantic top 5 relevant* | Keyword top 5 relevant* | Verdict |
|---|---|---|---|---|
| 1 | my parcel never showed up | 4 / 5 | 0 / 1 | Semantic much better |
| 2 | I can't log into my account | 3 / 5 | 1 / 5 | Semantic much better |
| 3 | the app is not working | 3 / 5 | 2 / 4 | Semantic better |
| 4 | cancel my booking | top-1 correct | 3-way tie | Semantic better (ranking) |
| 5 | How do I end my plan? | top-1 **wrong intent** | top-1 wrong intent | Both weak |
| 6 | turn off notifications | 0 / 5, **opposite intent** ranked high | 0 / 2 | Both fail, semantic fails "confidently" |
| 7 | I want to cancel my subscription | 2 / 5, opposite intent at #3 | 2 / 5 | Similar |

\*"Relevant" = the result would actually help a user with this request (my judgement).

---

## Cases where semantic search is clearly better

### 1. `"my parcel never showed up"` – different words, same meaning

| Semantic | Keyword |
|---|---|
| 1. doc_021 (0.568) My package was delivered to the wrong address. | 1. doc_023 (0.25) Can I pick up my order instead of having it shipped? |
| 2. doc_017 (0.519) I ordered an item, but it has not arrived yet. | *(nothing else)* |
| 3. doc_024 (0.512) The tracking status says delivered, but I received nothing. | |
| 4. doc_018 (0.480) Where can I track my package? | |

None of the relevant documents contain the words *parcel*, *never*, or *showed*. Keyword search
finds only one document, and only because of the word **"up"** (in "pick **up**"), which is a coincidental
match that has nothing to do with the request. Semantic search understands that *parcel ≈ package* and *never showed up ≈ has not
arrived / received nothing*, and returns the three best answers in its top 3.

### 2. `"I can't log into my account"` – synonyms and word forms

| Semantic | Keyword |
|---|---|
| 1. doc_010 (0.749) I cannot sign in even though my password is correct. | 1. doc_008 (0.333) Can I transfer my subscription to another account? |
| 2. doc_014 (0.708) My account was locked after several login attempts. | 2. doc_009 (0.333) I forgot my account password. |
| 3. doc_009 (0.584) I forgot my account password. | 3. doc_011 (0.333) How do I change the email address linked to my account? |
| | 4. doc_012 (0.333) I want to permanently delete my user account. |
| | 5. doc_013 (0.333) How can I create a new account? |

Keyword search can only match the generic word **"account"**, so all five results get the
same score (0.333). Their order only reflects their position in the dataset, not relevance. It misses the two best answers:
- doc_010 says *"cannot sign in"*: different words for *"can't log in"*.
- doc_014 says *"login"*: a different word form from *"log"*, and there is no stemming.

Semantic search puts exactly these two documents first.

### 3. `"the app is not working"` – ambiguous word senses

Keyword search returns doc_050 *"Where can I find a cafe suitable for **working** on a laptop?"*
(Leisure). It matches the word *working* but not its meaning. Semantic search keeps all five results in
Technical Support, with *"The app crashes immediately after I open it"* first.

### 4. `"cancel my booking"` – ranking, not just matching

Keyword search finds three documents containing *cancel*: a subscription, a shipment, and concert
tickets. All three score exactly 0.5 because *booking* ≠ *booked*, so it cannot tell which is best.
Semantic search ranks *"How can I cancel concert tickets I already booked?"* first with a clear margin
(0.730 vs. 0.525), because it understands that *booking* and *booked tickets* are related.

---

## Cases where semantic search does **not** match the user's intent

### A. `"turn off notifications"` – semantic search returns the opposite intent

| Semantic | Keyword |
|---|---|
| 1. doc_043 (0.503) I am **not receiving** notifications on my phone. | 1. doc_016 (0.667) How can I turn off two-factor authentication? |
| 2. doc_044 (0.369) How do I **enable** alerts for new orders? | 2. doc_043 (0.333) I am not receiving notifications on my phone. |
| 3. doc_016 (0.280) How can I turn off two-factor authentication? | |

The user wants **fewer** notifications, but the top two semantic results come from users who want
**more**:
- doc_043 complains that notifications do *not arrive*.
- doc_044 asks how to *enable* alerts.

The embedding captures the **topic** (notifications/alerts) very well, but largely ignores the
**direction** of the action (*turn off* vs. *enable*).

The dataset actually has no document about disabling notifications. The closest one is doc_060,
*"I want to stop receiving course recommendations"*. Semantic search still returns results with fairly
high scores and gives no signal that nothing relevant exists. Keyword search fails too, but at least
its #1 result shares the action *"turn off"*.

### B. `"How do I end my plan?"` – same topic word, different intent

| Semantic | Keyword |
|---|---|
| 1. doc_004 (0.414) Can I **switch to a cheaper** plan? | 1. doc_003 (0.5) I would like to **upgrade** to the premium subscription plan. |
| 2. doc_052 (0.321) How can I cancel **concert tickets** I already booked? | 2. doc_004 (0.5) Can I switch to a cheaper plan? |
| 3. doc_001 (0.317) I want to cancel my monthly subscription. ✅ | 3. doc_005 (0.5) My subscription renewed even though I asked to end it. |

The user wants to *end* (cancel) the plan, but semantic search ranks *"switch to a cheaper plan"*
first. That document shares the word *plan* and the subscription topic, but its intent is to *keep*
the service. The first relevant answer (doc_001) is only #3 and comes after a result about concert tickets. Note that the
scores are low (0.41 at most): the query is short and uses a different word (*plan* vs. *subscription*)
than the right answer.

### C. `"I want to cancel my subscription"` – the example from the assignment

Semantic search finds the right answer first (doc_001, 0.941). However, #3 is
*"How do I **reactivate** an expired subscription?"* (0.684), which is the opposite action. Its score is almost identical to
#2 (0.688), which is relevant. Everything about *subscriptions* scores high, regardless of whether the user wants to cancel,
pause, or reactivate.

---

## Conclusions

1. **Semantic search wins when users phrase things differently from the documents.** It handles
   synonyms (*parcel/package*, *log in/sign in*), different word forms (*login/log*,
   *booking/booked*), and paraphrases (*never showed up / has not arrived*). Keyword search either finds
   nothing or matches irrelevant documents through incidental words (*up*, *working*).
2. **Keyword search cannot rank.** When several documents share the same keyword they get identical
   scores, so the order carries no information (queries 2 and 4).
3. **Semantic search captures *topic* better than *intent*.** Documents about the same subject get
   high similarity even when the requested action is the opposite (*turn off* vs. *enable*, *cancel*
   vs. *reactivate/switch*). This is the main weakness that Part D tries to fix.
4. **Semantic search always returns something.** It never returns "no results", even when nothing
   relevant exists (query 6). The absolute score is only a weak relevance signal: 0.50 was the top score
   for an irrelevant result in query 6, while 0.48 was a relevant result in query 1.
5. **The two methods fail differently,** which suggests that combining them (hybrid search) could
   help. For example, keyword search kept the action words *"turn off"* that semantic search
   under-weighted. (Part D tried a simple hybrid score: it barely helped with intent, so I chose
   reranking instead. See `part_d_analysis.md`.)
