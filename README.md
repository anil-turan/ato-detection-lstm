# Account Takeover (ATO) Detection Through Behavioural Analytics

Individual sub-topic of **Group 6 – Real-Time Fraud Detection for Digital Payments**
(University of Greenwich, MSc Data Science, COMP1889).

This repo detects likely **account-takeover** transactions in the IEEE-CIS Fraud
Detection dataset using a sequence model (LSTM) over each card's recent
transaction history, compared against Random Forest and Isolation Forest
baselines.

---

## 1. What this project does

The IEEE-CIS dataset only has a general `isFraud` flag, not an ATO label. So we
build a **proxy ATO label** from behavioural warning signs, then train models to
predict it from each card's last 10 transactions.

There are two versions of the pipeline in this repo:

| Version | Where | Status |
|---|---|---|
| Original | `notebooks/01..06` | First attempt. Has a data-leakage problem (see below). |
| **v2 (leak-free)** | `src/` | Clean rebuild. **Use this one.** |

The `src/` pipeline is the corrected, reproducible version and is the basis for
the reported results.

---

## 2. Quick start

```bash
# 1. install dependencies
python3 -m pip install -r requirements.txt

# 2. run the whole leak-free pipeline (labels -> features -> train -> evaluate)
PYTHONPATH=. python3 -m src.run_pipeline

# (use --rebuild to force the labelling and feature steps to run again)
PYTHONPATH=. python3 -m src.run_pipeline --rebuild
```

Raw data is expected in `data/raw/` (the five IEEE-CIS CSV files) and the merged
parquet files in `data/processed/`.

---

## 3. Pipeline steps (the `src/` modules)

| Module | Job |
|---|---|
| `src/config.py` | All paths, the random seed, the window size, and the list of label-only columns. |
| `src/data/labeling.py` | Builds the 3 behavioural signals and the `ato_proxy` label. |
| `src/features/build.py` | Leak-free features, **card-level** train/test split, train-only scaling, sliding-window sequences. |
| `src/models/lstm.py` | The LSTM model and its training loop (class weights for imbalance). |
| `src/models/baselines.py` | Random Forest + Isolation Forest baselines (balanced). |
| `src/evaluation/metrics.py` | ROC-AUC, PR-AUC, threshold sweep, recall-target threshold, plots. |
| `src/run_pipeline.py` | Runs everything end to end and writes the results. |

Outputs are written to `outputs/`:
`outputs/models/lstm_v2_*.keras`, `outputs/figures/*_v2_*.png`,
`outputs/reports/evaluation_v2.json`, `outputs/reports/results_table_v2.md`.

---

## 4. The proxy ATO label

A transaction is labelled `ato_proxy = 1` when it is **fraud** AND shows **at
least 2 of 3** behavioural signals:

1. **Device/identity change** – the device changed for this card, or a
   proxy/VPN network was used.
2. **Session anomaly** – an id_01..id_11 value is more than 2 standard
   deviations from the card's own history.
3. **Velocity anomaly** – two or more transactions on the same card within 2 hours.

This gives **2,362 positives out of 590,540 rows (0.40%)** – a very imbalanced
problem.

> The signal columns are used **only to build the label**. They are listed in
> `config.LABEL_ONLY_COLUMNS` and removed before training, so the model never
> sees them.

---

## 5. Results (leak-free pipeline)

Test set = **118,963 sequences (442 positives)**, split **by card** so no card
appears in both train and test.

All numbers below are **real**, from `outputs/reports/evaluation_v2.json` (LSTM +
baselines) produced by `PYTHONPATH=. python3 -m src.run_pipeline`.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| LSTM (ours) | 0.905 | **0.0275** |
| Random Forest | **0.917** | 0.0269 |
| Isolation Forest | 0.863 | 0.0183 |

**Why ROC-AUC looks good but the task is still hard.** The ROC-AUC values look
strong (~0.91), but the **PR-AUC is only ~0.027** — that is the honest picture for
a 0.4%-positive problem, and the reason we report PR-AUC alongside ROC-AUC. The
LSTM's recall/precision trade-off makes it concrete (real numbers, same JSON):

| Operating point | Threshold | Recall | Precision |
|---|---|---|---|
| Best F1 | 0.95 | 0.07 | 0.049 |
| Catch ~90% of attacks | 0.45 | 0.90 | 0.019 |

Catching ~90% of attacks means **~2% precision** — the model flags many sessions
to find most attacks. This is the core finding: a single hard threshold is not
usable, which is why the two-tier policy (allow / step-up auth / block) is
proposed in `docs/METHODOLOGY_AND_LIMITATIONS.md`.

**LSTM vs Random Forest.** After removing the leakage the two are essentially
tied (LSTM ROC-AUC 0.905 vs RF 0.917; LSTM PR-AUC 0.0275 slightly ahead of RF
0.0269). The LSTM does not dominate — an honest, expected result given that most
cards have very few transactions, so the 10-step sequences are mostly padding.

Key context:

- The earlier notebook reported **AUC 0.944**, but that score was **inflated by
  data leakage** (label-defining columns `signal_count` / `time_since_prev` were
  also used as features, and the split was random over rows of the same card).
  The clean pipeline removes both, and the fair LSTM score is **0.905** — close
  to the leaked figure here, but now trustworthy.
- Because attacks are so rare (~0.4%), high recall will come at very low
  precision, so a single hard threshold is not usable. See
  `docs/METHODOLOGY_AND_LIMITATIONS.md` for the two-tier operating policy
  (allow / step-up auth / block) and the honest discussion of the "behavioural
  biometrics" wording vs the features the dataset actually allows.

## 7. Environment note (TensorFlow)

The LSTM was trained with TensorFlow 2.21 on an Apple Silicon (M1) Mac. The
sklearn baselines and the data build do **not** need TensorFlow, so they run
anywhere (`src.evaluation.run_baselines`, `src.evaluation.explain`,
`src.inference.predict`). TensorFlow is imported lazily inside `src/models/lstm.py`,
so the non-LSTM parts still work even if TensorFlow is missing.

---

## 6. Reproducibility

- One random seed (`config.RANDOM_SEED = 42`) is used for numpy, scikit-learn and
  TensorFlow.
- The scaler is fit on training rows only and saved to `outputs/scaler_v2.pkl`.
- Imputation medians (train only) are saved to `outputs/imputation_medians_v2.csv`.
- Pinned dependency versions are in `requirements.txt`.
