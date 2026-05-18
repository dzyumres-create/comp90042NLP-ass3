# comp90042NLP-ass3

## Structure of Report (if you want to write some parts, don't forget the number. for example: **3.1** Data Analysis ):

## Structure
1. Abstract 
2. Introduction 
3. Approach (our whole system process without analysis)

Please attach the pictures and tables to make content more vivid. 

3.1 Data Analysis — label distribution, class imbalance, evidence length 
3.2 Preprocessing — lemmatisation, why it matters for BM25
3.3 Evidence Retrieval — BM25, grid search, why recall over F1 , hyper parameter tradeoff
3.4 Evidence Reranking — cross-encoder, domain mismatch, fine-tuning rationale 
3.5 Claim Classification — BERT, [EVID_SEP], noise training C1, class weights C2 
3.6 Alternative approaches considered — bi-encoder considered and rejected, why 

4. Experiments

4.1 Experimental setup — dataset stats, hardware, hyperparameters for all models (Teammate B for classifier, You for reranker, Teammate A for BM25)
4.2 Evaluation metrics — F, A, H_FA definitions 

5. Results

5.1 Main ablation table — every configuration row 
5.2 Retrieval analysis — BM25 recall at different top-k 
5.3 Pipeline improvement analysis — what each component added 
5.4 Novelty experiments — PL1 ensemble sweep table + N1 abstention table + analysis of why both failed 

6. Conclusion 
7. Team Contributions 
8. References 

# 3.1 data Analysis

## Key statistics for the report

## Input files(raw files)

| File | What it is | Used by |
|---|---|---|
| `train-claims.json` | 1,228 labelled claims with `claim_text`, `claim_label`, and gold `evidences` IDs | Classification (training), oracle experiment |
| `dev-claims.json` | ~300 labelled claims, same format as train | evaluation |
| `test-claims-unlabelled.json` | Unlabelled claims, only `claim_text` | Final leaderboard submission |
| `evidence.json` | 1.2M evidence passages, `{evidence_id: raw_text}` | Retrieval (Transformer reranker), classifier |
| `dev-claims-baseline.json` | Example prediction output from a fake baseline | Reference only — shows the required output format for `eval.py` |
| `eval.py` | Evaluation script | Run this to get F-score, accuracy, and harmonic mean on dev |

---

| Statistic | Value |
|---|---|
| Train claims | *1228* |
| Dev claims | *154* |
| Evidence passages | *1208827* |
| Mean claim length | *20.1* a words |
| Mean gold evidences per claim | *3.4* |
| SUPPORTS (train) | *42.2%* |
| REFUTES (train) | *16.2%* |
| NOT_ENOUGH_INFO (train) | *31.4%* |
| DISPUTED (train) | *10.1%* |
| SUPPORTS (dev) | *44.2%* |
| REFUTES (dev) | *17.5%* |
| NOT_ENOUGH_INFO (dev) | *26.6%* |
| DISPUTED (dev) | *11.7%* |

> Most evidence passages fall in the 11–50 word range. There is also a large number of very short passages (≤5 words) which may pose challenges for retrieval.


# 3.2 Preprocessing

Before retrieval, all claim texts and evidence passages are passed through a shared preprocessing pipeline implemented in the clean_text() function. The same function is applied uniformly to both claims and the full evidence corpus of 1,208,827 passages, ensuring that the vocabulary space seen during retrieval is consistent between query and document representations.
The pipeline applies the following steps in order:

1. **Contraction expansion** Contractions such as isn't and it's are expanded to their full forms using the contractions library, preventing the same word from appearing as two different tokens depending on whether it was contracted.
2. **Lowercasing**  All text is converted to lowercase to eliminate case-based token mismatches (e.g., CO2 and co2 being treated as distinct terms).
3. **Punctuation and special character removal** All non-alphanumeric characters are replaced with spaces using a regular expression, removing symbols, hyphens, and citation artefacts common in scientific text.
4. **Tokenisation** Tokens are produced using NLTK's word_tokenize.
5. **Stopword removal** Standard English stopwords from NLTK's stopword list are removed, along with any single-character tokens, to reduce noise and index size.
6. **POS-aware** lemmatisation. Each remaining token is part-of-speech tagged using NLTK's averaged perceptron tagger, and the tag is mapped to WordNet's four-category POS scheme (noun, verb, adjective, adverb). The WordNetLemmatizer then reduces each token to its base form using the correct POS context — for example, warming is correctly lemmatised to warm (verb) rather than warming (noun). This is an improvement over stemming, which can produce non-words, and over POS-agnostic lemmatisation, which defaults all tokens to the noun category and produces incorrect results for verbs and adjectives.
---

## Output files (preprocessed, ready to load)

All saved to the `preprocessed/` folder.

| File | What it contains | Format | which parts load it |
|---|---|---|---|
| `train_clean.json` | All train claims with an added `clean_tokens` field (list of tokens) | Same as `train-claims.json` plus `clean_tokens` per entry | Retrieval (BM25 stage) |
| `dev_clean.json` | Same as above for dev claims | Same structure | Retrieval (BM25 stage) |
| `test_clean.json` | Same as above for test claims | Same structure | Retrieval (BM25 stage) |
| `evidence_clean.json` | All 1.2M evidence passages cleaned into token lists | `{evidence_id: [token, token, ...]}` | Retrieval (BM25 stage only) |
| `stats_summary.json` | Key dataset statistics | JSON dict | Reference for report writing |
| `class_distribution.png` | Bar chart of label counts in train and dev | Image | Report Section 2 figure |

---

### Retrieval

#### BM25 first stage

Load `evidence_clean.json`. Each value is already a list of tokens — pass directly into `rank_bm25` without any further processing.

```python
import json

with open('preprocessed/evidence_clean.json', 'r') as f:
    evidence_clean = json.load(f)

# evidence_clean['evidence-0'] → ['john', 'bennet', 'lawes', 'english', ...]
```

Load `train_clean.json` and use the `clean_tokens` field for the claim side of BM25.

```python
with open('preprocessed/train_clean.json', 'r') as f:
    train_clean = json.load(f)

# train_clean['claim-2967']['clean_tokens'] → ['south', 'australia', 'expens', ...]
```

# 3.4 Evidence Reranking

## 3.4.1 Model Choice: Cross-Encoder
Given the top-20 candidate passages returned by BM25, the reranking stage re-scores each candidate against the claim and selects the top-3 for downstream classification. We use a cross-encoder architecture based on cross-encoder/ms-marco-MiniLM-L-6-v2 [CITE: Reimers & Gurevych, 2019].
A cross-encoder processes a claim-evidence pair by concatenating the two texts with a [SEP] separator and passing the full sequence through a transformer encoder:
            **score(claim, evidence) = sigmoid( Transformer( "claim text [SEP] evidence text" ) )**
This joint encoding allows every attention head in the transformer to attend across both texts simultaneously, capturing semantic relationships that surface-form matching cannot — for example, recognising that "carbon dioxide" and "CO₂" refer to the same entity, or that a passage discussing "anthropogenic warming" is relevant to a claim about "human-caused climate change".
This is in contrast to a bi-encoder, which encodes the claim and each evidence passage independently and computes similarity via dot product or cosine distance. While bi-encoders are faster at inference, they cannot model cross-text interactions, which makes them poorly suited to a task where relevance depends on nuanced semantic alignment between a scientific claim and a passage (see Section 3.6 for further discussion of this design decision).
## 3.4.2 Domain Mismatch and the Motivation for Fine-Tuning
The ms-marco model was pre-trained on the Microsoft MARCO dataset — a large collection of web search queries and web page passages. Climate science claims are structurally and semantically different from web search queries in several important ways: they are longer and more complex, they assert specific scientific propositions, and they rely on specialised vocabulary that does not appear in web corpora. Terms such as radiative forcing, equilibrium climate sensitivity, and paleoclimate proxy are routine in our evidence corpus but absent from the pre-training distribution.
As a result, using the ms-marco model in a zero-shot setting introduces a domain mismatch: the model may rank evidence passages that share common surface-level words with the claim above passages that are scientifically more relevant. The off-the-shelf reranker achieved H_FA = 0.243 on the development set, which, while an improvement over BM25 alone, leaves significant headroom.
## 3.4.3 Fine-Tuning Procedure
To adapt the reranker to the climate domain, we fine-tuned it on claim-evidence pairs constructed from the training set. The training data was built as follows:

**Positive pairs**: each gold evidence passage linked to a claim in train-claims.json was paired with that claim and labelled 1.
**Negative pairs**: the remaining passages from BM25's top-20 candidates for that claim (those not in the gold evidence set) were labelled 0. 
To control the class imbalance, a maximum negative-to-positive ratio of 5:1 was enforced, with negatives sampled randomly when the ratio would otherwise be exceeded.

This construction strategy uses hard negatives — passages that BM25 considered plausible but that are not truly relevant. Hard negatives are more informative for training than randomly sampled passages, because they force the model to learn fine-grained distinctions rather than trivial ones. The model was fine-tuned for 3 epochs with a batch size of 16, using a warmup of 10% of total training steps. The fine-tuned weights were saved and used for all subsequent reranking in the pipeline.
## 3.4.4 Effect of Fine-Tuning
Fine-tuning produced a meaningful improvement across all metrics on the development set. H_FA increased from 0.243 (off-the-shelf) to 0.279 (fine-tuned), a relative gain of approximately 15%. This confirms that domain adaptation is effective even with a relatively small training set (1,228 claims), and that the vocabulary and structural differences between web search and climate science are large enough to warrant explicit adaptation.
The value of fine-tuning is further demonstrated by the negative result from our ensemble experiment (Section 5.4): adding BM25's raw top-k passages alongside the reranker's output did not improve performance, suggesting that the fine-tuned reranker had already internalised the lexical matching capabilities that the ensemble was intended to supplement.

---

# 3.6 Alternative Approaches Considered

## 3.6.1 Bi-Encoder for Reranking
Before adopting the cross-encoder, we considered a bi-encoder for reranking, which encodes the claim and each evidence passage independently and computes relevance via cosine similarity between the two resulting vectors. While bi-encoders offer inference speed advantages at scale, this is irrelevant at our reranking stage where only 20 candidates need scoring. More critically, independent encoding prevents the model from capturing cross-text semantic interactions — the kind of joint reasoning required to determine whether a passage is genuinely relevant to a specific scientific claim, rather than merely sharing surface vocabulary. The cross-encoder was therefore preferred.
## 3.6.2 Dense Retrieval as a BM25 Replacement
We also considered replacing BM25 with a dense retrieval model such as DPR [CITE: Karpukhin et al., 2020] for initial candidate generation. This was rejected on practical grounds: encoding and indexing 1,208,827 passages into a dense vector index exceeded our available GPU resources. Given that BM25 at top-20 already achieves a recall of approximately 0.36 on the development set, the cost-benefit case for dense retrieval was not compelling.

---

# Classification 


---

# 5.3 Pipeline Improvement Analysis

Figure 1 shows the full four-stage architecture of our system. Each stage builds on the previous, and the contribution of each component can be quantified by comparing performance before and after its introduction.
***Stage 1 — Preprocessing establishes*** a consistent token vocabulary across claims and the 1,208,827-passage evidence corpus. By applying contraction expansion, lowercasing, stopword removal, and POS-aware lemmatisation, the pipeline reduces vocabulary mismatch that would otherwise penalise BM25 term overlap. The impact of this stage is embedded in all downstream scores and cannot be isolated without a controlled ablation, but it is a necessary foundation for the retrieval recall figures reported below.
***Stage 2 — BM25 Retrieval*** achieves a recall of approximately 0.36 at top-20 on the development set. This is the ceiling that the reranker and classifier must work within — any gold evidence passage not retrieved at this stage is unrecoverable. The grid-searched hyperparameters (k1, b) were optimised specifically for recall rather than F-score, reflecting the asymmetric cost of missed evidence versus extra candidates.
***Stage 3 — Cross-encoder Reranking*** is where the most significant improvement occurs. The off-the-shelf ms-marco reranker brought H_FA to 0.243. After fine-tuning on climate claim-evidence pairs, H_FA improved to 0.279 — a 15% relative gain. This confirms that domain adaptation is the primary driver of reranking quality, and that the vocabulary mismatch between web search pre-training and climate science text is large enough to be a measurable bottleneck.
***Stage 4 — BERT Classification*** achieves a classification accuracy of A = 0.519 on the development set, with F = 0.191 (retrieval F-score is fixed by the reranker at this point). The key design choice here — training on reranker-predicted evidence rather than gold evidence — addresses the train/test distribution mismatch that would otherwise cause the classifier to overfit to clean, perfectly-retrieved inputs it will never see at inference time. The final H_FA of 0.279 represents the combined effect of all four stages working together.
Figure 1: System architecture of the climate fact-checking pipeline. Metrics shown on the right reflect development set performance at each stage.

---

# 5.4 Novelty Experiments
We conducted two novelty experiments beyond the core pipeline. Both produced negative results — performance did not improve — but the analyses reveal important properties of the system and contribute to the understanding of where the current bottlenecks lie.
## 5.4.1 Experiment PL1 — BM25 + Reranker Ensemble Sweep
Hypothesis. The fine-tuned reranker, while strong at semantic ranking, might occasionally demote a lexically-relevant passage that BM25 ranked highly. An ensemble combining BM25's raw top-k picks with the reranker's top-1 pick could recover such cases and improve retrieval coverage.
Setup. We fixed the reranker's top-1 evidence passage and supplemented it with BM25's top-k raw candidates (k = 1, 2, 3), producing evidence sets of size 2, 3, and 4 respectively. Each configuration was passed to the classifier and evaluated on the development set.
Results.

|Configuration | F | A | H_FA |
| -------- | -------- | -------- | -------- |
| BM25 top-1 + reranker top-1    | 0.181     | 0.506     | 0.26     |
| BM25 top-2 + reranker top-1    | 0.173     | 0.506     | 0.258    |
| BM25 top-3 + reranker top-1    | 0.166     | 0.500     | 0.249     |
| Pure reranker top-3 (final system)    | 0.191     | 0.519     | 0.279     |
Table 2: Ensemble sweep results on the development set. Pure reranker top-3 dominates across all metrics.
Analysis. The ensemble consistently underperforms the pure reranker configuration across all metrics and all values of k. Moreover, performance degrades monotonically as more BM25 passages are added, suggesting that BM25's raw candidates are introducing noise rather than providing complementary signal. This result indicates that the fine-tuned reranker has already internalised the vocabulary-matching capability that the ensemble was intended to compensate for externally — domain fine-tuning made the ensemble redundant. This contrasts with Group 21's finding [CITE: Group 21 report], where an ensemble improved performance, likely because their reranker was trained from scratch without a pre-trained initialisation and therefore benefited from BM25's lexical signal.
5.4.2 Experiment N1 — Confidence-Based Abstention Sweep
Hypothesis. When the reranker assigns a low maximum confidence score to all 20 candidates for a given claim, it may indicate that none of the retrieved passages are genuinely relevant. In such cases, overriding the classifier with a NOT_ENOUGH_INFO label might be more accurate than trusting an unreliable classification.
Setup. We introduced an abstention threshold τ. For any claim where the reranker's highest sigmoid score across all 20 candidates fell below τ, the system output NOT_ENOUGH_INFO regardless of the classifier's prediction. We swept τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7} and recorded H_FA, F, A, and the proportion of claims forced to NOT_ENOUGH_INFO (NEI rate).
Results.
τFAH_FANEI rate0.10.1910.4090.26061.7%0.20.1910.4160.26263.6%0.30.1910.4290.26464.9%0.40.1910.4090.26066.9%0.50.1910.4090.26066.9%0.60.1910.3830.25569.5%0.70.1910.3770.25370.1%Baseline (no abstention)0.1910.5200.2790%
Table 3: Confidence-based abstention sweep on the development set. Baseline (no abstention) achieves the best H_FA at every threshold.
Analysis. Abstention consistently hurts performance at every threshold tested. Two observations explain this result. First, the NEI rate is extremely high even at the lowest threshold (τ = 0.1 forces 61.7% of claims to NOT_ENOUGH_INFO), revealing that the reranker's raw sigmoid scores are not well-calibrated as absolute confidence values. The model was trained to rank 20 candidates relative to each other, not to produce scores that are meaningful in absolute terms — a low score does not reliably indicate that no relevant evidence exists. Second, accuracy degrades steadily as τ increases because the system incorrectly overrides correct classifier predictions for claims that did have good evidence but happened to receive low reranker scores. The retrieval F-score (F = 0.191) is unaffected across all rows, as abstention only changes the label, not the evidence set.




---

# Reference

[CITE: Reimers & Gurevych, 2019] — the sentence-transformers / cross-encoder paper
[CITE: Nogueira & Cho, 2019] — ms-marco model
[CITE: Nguyen et al., 2016] — the Microsoft MARCO dataset itself

   
