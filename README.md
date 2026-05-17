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

## What this document covers

This document describes everything done in the preprocessing stage, what files were produced, and which files should load.

---

## What preprocessing does and why

The raw data files cannot be fed directly into BM25 retrieval — they contain contractions, punctuation, stopwords, and mixed casing that hurt keyword matching. Preprocessing cleans the text into a consistent format for BM25.

At the same time, the raw original files are preserved untouched for the Transformer reranker and classifier, which handle their own tokenisation internally.

---

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

## Cleaning steps applied

These steps are applied **in order** to both claim texts and evidence passages:

1. **Expand contractions** — *don't → do not*, *it's → it is*. 
2. **Lowercase** — *Antarctica → antarctica*. 
3. **Remove punctuation and special characters** — replaces anything that is not a letter, number, or space with a space.
4. **Tokenise** — splits the string into individual word tokens.
5. **Remove stopwords** — removes common words (*the, is, at, a, an...*) that appear everywhere and carry no retrieval signal.
6. **Remove single-character tokens** — removes leftover noise like lone letters or digits.
7. **Output is a list of tokens** — not a string. This is the format `rank_bm25` expects directly.

> **Note:** Stemming was deliberately skipped. BM25 handles term frequency internally and stemming can merge words that should stay distinct.

> **Note:** The `contractions` library crashes on very short or unusual strings in the evidence corpus. The cleaning function wraps that step in a `try/except` so it skips gracefully without stopping the pipeline.

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

#### Transformer reranker

Load the original `evidence.json` directly — raw text, no preprocessing needed. The model's own tokenizer handles everything.

```python
with open('data/evidence.json', 'r') as f:
    evidence_raw = json.load(f)

# evidence_raw['evidence-0'] → 'John Bennet Lawes, English entrepreneur...'
```

---

### Classification 


---



---


   
