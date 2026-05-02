# comp90042NLP-ass3
# Preprocessing Handoff — COMP90042

## What this document covers

This document describes everything done in the preprocessing stage, what files were produced, and what each teammate should load for their component.

---

## What preprocessing does and why

The raw data files cannot be fed directly into BM25 retrieval — they contain contractions, punctuation, stopwords, and mixed casing that hurt keyword matching. Preprocessing cleans the text into a consistent format for BM25.

At the same time, the raw original files are preserved untouched for the Transformer reranker and classifier, which handle their own tokenisation internally.

---

## Input files (raw, from the project)

| File | What it is | Used by |
|---|---|---|
| `train-claims.json` | 1,228 labelled claims with `claim_text`, `claim_label`, and gold `evidences` IDs | Classification teammate (training), oracle experiment |
| `dev-claims.json` | ~300 labelled claims, same format as train | Everyone for evaluation |
| `test-claims-unlabelled.json` | Unlabelled claims, only `claim_text` | Final leaderboard submission |
| `evidence.json` | 1.2M evidence passages, `{evidence_id: raw_text}` | Retrieval teammate (Transformer reranker), classifier |
| `dev-claims-baseline.json` | Example prediction output from a fake baseline | Reference only — shows the required output format for `eval.py` |
| `eval.py` | Evaluation script | Run this to get F-score, accuracy, and harmonic mean on dev |

---

## Cleaning steps applied

These steps are applied **in order** to both claim texts and evidence passages:

1. **Expand contractions** — *don't → do not*, *it's → it is*. Ensures BM25 sees full word forms rather than contracted tokens.
2. **Lowercase** — *Antarctica → antarctica*. BM25 is case-sensitive so this prevents duplicate matching.
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

| File | What it contains | Format | Who loads it |
|---|---|---|---|
| `train_clean.json` | All train claims with an added `clean_tokens` field (list of tokens) | Same as `train-claims.json` plus `clean_tokens` per entry | Retrieval teammate (BM25 stage) |
| `dev_clean.json` | Same as above for dev claims | Same structure | Retrieval teammate (BM25 stage) |
| `test_clean.json` | Same as above for test claims | Same structure | Retrieval teammate (BM25 stage) |
| `evidence_clean.json` | All 1.2M evidence passages cleaned into token lists | `{evidence_id: [token, token, ...]}` | Retrieval teammate (BM25 stage only) |
| `stats_summary.json` | Key dataset statistics | JSON dict | Reference for report writing |
| `class_distribution.png` | Bar chart of label counts in train and dev | Image | Report Section 2 figure |

---

## What each teammate should load

### Retrieval teammate

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

### Classification teammate

Load the original `train-claims.json` and `dev-claims.json` directly — raw text, your model's tokenizer handles it. Use the gold `evidences` IDs during training to look up the actual passage text from `evidence.json`.

```python
import json

with open('data/train-claims.json', 'r') as f:
    train_claims = json.load(f)

with open('data/evidence.json', 'r') as f:
    evidence = json.load(f)

# For a given claim, get its gold evidence texts:
entry = train_claims['claim-2967']
gold_texts = [evidence[eid] for eid in entry['evidences']]
```

> **Important:** Train on gold evidence from `train-claims.json`. At inference time you will receive retrieved evidence IDs from the retrieval teammate instead. The gap between these two is worth discussing in the report.

---

## Oracle experiment result

The oracle experiment trains a simple logistic regression classifier on gold evidence and evaluates on dev. This number represents the **upper bound** on classification accuracy — what the system could achieve if retrieval were perfect.

| Experiment | Dev Accuracy |
|---|---|
| Oracle (gold evidence + logistic regression) | *(fill in your number after running Cell 9)* |

If the final system's accuracy is far below this, the bottleneck is **retrieval quality**.  
If it is close, the bottleneck is the **classifier itself**.  
Share this number with both teammates.

---

## Key statistics for the report

Fill these in from `stats_summary.json` after running the notebook:

| Statistic | Value |
|---|---|
| Train claims | *(number)* |
| Dev claims | *(number)* |
| Evidence passages | *(number)* |
| Mean claim length | *(number)* words |
| Mean gold evidences per claim | *(number)* |
| SUPPORTS (train) | *(%)* |
| REFUTES (train) | *(%)* |
| NOT_ENOUGH_INFO (train) | *(%)* |
| DISPUTED (train) | *(%)* |

> Most evidence passages fall in the 11–50 word range. There is also a large number of very short passages (≤5 words) which may pose challenges for retrieval — the retrieval teammate should be aware of this.

---

## One thing to be aware of

Colab's local storage resets when the session ends. Download the preprocessed files and store them somewhere persistent (Google Drive or your own machine) after running the notebook.

**Do not re-run the evidence cleaning cell unnecessarily** — it processes 1.2M passages and takes several minutes.

   
