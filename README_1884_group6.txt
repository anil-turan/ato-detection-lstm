================================================================================
COMP1884 GROUP PROJECT - GROUP 6
Real-Time Fraud Detection for Digital Payments
Sub-topic (this folder): Account Takeover (ATO) Detection Through Behavioural
                         Analytics - Anil Turan (Banner ID 001455222)
================================================================================

This README explains how to access / re-create the data sources and how to run
the code in this folder. It is the file required by the COMP1884 report
specification (README_1884_groupX.txt).


--------------------------------------------------------------------------------
1. WHAT THIS CODE DOES
--------------------------------------------------------------------------------
It detects account takeover (ATO) on digital payment data. ATO is when an
attacker uses stolen credentials to access a genuine account. Because the public
dataset has no ATO label, the code builds a behavioural proxy label, trains an
LSTM that reads each account's last 10 transactions as a sequence, compares it
against Random Forest and Isolation Forest baselines, explains the model with
SHAP, and turns the score into an allow / step-up / block decision policy.

Final, leak-free results (test = 118,963 sequences, 442 ATO cases):
   LSTM (ours)        ROC-AUC 0.905   PR-AUC 0.0275
   Random Forest      ROC-AUC 0.917   PR-AUC 0.0269
   Isolation Forest   ROC-AUC 0.863   PR-AUC 0.0183


--------------------------------------------------------------------------------
2. DATA SOURCE - HOW TO ACCESS / RE-CREATE IT
--------------------------------------------------------------------------------
Source : IEEE-CIS Fraud Detection dataset (Kaggle competition, 2019).
URL    : https://www.kaggle.com/c/ieee-fraud-detection/data
Licence: Kaggle competition rules. The data is anonymised and public. No
         personally identifiable information (PII) is present or used.

The raw CSV files (train_transaction.csv, train_identity.csv, etc.) are about
1.3 GB in total, so they are NOT included in this submission. To re-create the
data locally:

   1. Download the files from the Kaggle URL above (free account required).
   2. Place them in:   data/raw/
        data/raw/train_transaction.csv
        data/raw/train_identity.csv
        data/raw/test_transaction.csv
        data/raw/test_identity.csv

A SMALL REPRESENTATIVE SAMPLE is provided so the work can be inspected without
the full download:
        data/sample/df_sample_5000.csv   (first 5,000 merged rows)

The single processed input the pipeline actually reads is created from the raw
files and is also large, so it is not shipped. The pipeline regenerates it.


--------------------------------------------------------------------------------
3. HOW TO RUN
--------------------------------------------------------------------------------
Environment:
   Python 3.11. Install dependencies with:
        pip install -r requirements.txt
   (TensorFlow is only needed for the LSTM. The data build and the sklearn
    baselines run without it.)

Run the whole pipeline (labelling -> features -> LSTM -> baselines -> evaluation),
from inside this folder:
        PYTHONPATH=. python3 -m src.run_pipeline

Useful individual steps:
        PYTHONPATH=. python3 -m src.evaluation.run_baselines   # sklearn only, no TF
        PYTHONPATH=. python3 -m src.evaluation.explain          # SHAP feature importance
        PYTHONPATH=. python3 -m src.inference.predict           # writes the dashboard CSV

Outputs are written to:
        outputs/reports/   (evaluation_v2.json, results_table_v2.md, scores CSV)
        outputs/figures/   (ROC, PR and SHAP charts, *_v2.png)
        outputs/models/    (trained LSTM, *.keras)


--------------------------------------------------------------------------------
4. CODE LAYOUT
--------------------------------------------------------------------------------
   src/config.py                 settings, random seed, leak-free column list
   src/data/labeling.py          builds the ATO proxy label (3 signals)
   src/features/build.py         leak-free features, card-based split, sequences
   src/models/lstm.py            LSTM model (imports TensorFlow lazily)
   src/models/baselines.py       Random Forest + Isolation Forest
   src/evaluation/metrics.py     ROC-AUC, PR-AUC, threshold analysis, charts
   src/evaluation/run_baselines.py   baseline-only evaluation
   src/evaluation/explain.py     SHAP explainability
   src/inference/predict.py      0-100 risk score + allow/step-up/block policy
   src/run_pipeline.py           runs everything end to end
   docs/METHODOLOGY_AND_LIMITATIONS.md   full method + honest limitations
   README.md                     detailed project overview

GitHub (private): https://github.com/anil-turan/ato-detection-lstm


--------------------------------------------------------------------------------
5. IMPORTANT NOTES ON HONESTY AND VALIDITY
--------------------------------------------------------------------------------
 - An earlier notebook version reported ROC-AUC 0.944. That number was inflated
   by DATA LEAKAGE (label-defining columns used as features, and a random row
   split that mixed the same account across train and test). The pipeline in
   src/ fixes both, which is why the honest LSTM score is 0.905.
 - The dataset is a TRANSACTION dataset, not a session/biometric one. Features
   such as typing cadence do not exist in it; the code uses the closest proxies
   (device change, velocity, amount deviation, account age, id_* fields).
 - The old notebooks (notebooks/) are kept only to show the project's history.
   They contain the leaky earlier version. The trustworthy system is src/.

================================================================================
Contact: Anil Turan, Group 6, COMP1884, University of Greenwich.
================================================================================
