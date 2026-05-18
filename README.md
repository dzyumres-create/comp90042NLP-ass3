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
3.4.3 Fine-Tuning Procedure
To adapt the reranker to the climate domain, we fine-tuned it on claim-evidence pairs constructed from the training set. The training data was built as follows:

**Positive pairs**: each gold evidence passage linked to a claim in train-claims.json was paired with that claim and labelled 1.
**Negative pairs**: the remaining passages from BM25's top-20 candidates for that claim (those not in the gold evidence set) were labelled 0. To control the class imbalance, a maximum negative-to-positive ratio of 5:1 was enforced, with negatives sampled randomly when the ratio would otherwise be exceeded.

This construction strategy uses hard negatives — passages that BM25 considered plausible but that are not truly relevant. Hard negatives are more informative for training than randomly sampled passages, because they force the model to learn fine-grained distinctions rather than trivial ones. The model was fine-tuned for 3 epochs with a batch size of 16, using a warmup of 10% of total training steps. The fine-tuned weights were saved and used for all subsequent reranking in the pipeline.
## 3.4.4 Effect of Fine-Tuning
Fine-tuning produced a meaningful improvement across all metrics on the development set. H_FA increased from 0.243 (off-the-shelf) to 0.279 (fine-tuned), a relative gain of approximately 15%. This confirms that domain adaptation is effective even with a relatively small training set (1,228 claims), and that the vocabulary and structural differences between web search and climate science are large enough to warrant explicit adaptation.
The value of fine-tuning is further demonstrated by the negative result from our ensemble experiment (Section 5.4): adding BM25's raw top-k passages alongside the reranker's output did not improve performance, suggesting that the fine-tuned reranker had already internalised the lexical matching capabilities that the ensemble was intended to supplement.

---

### Classification 


---



---
# Reference

[CITE: Reimers & Gurevych, 2019] — the sentence-transformers / cross-encoder paper
[CITE: Nogueira & Cho, 2019] — ms-marco model
[CITE: Nguyen et al., 2016] — the Microsoft MARCO dataset itself

   
