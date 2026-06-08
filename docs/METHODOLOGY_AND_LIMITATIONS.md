# Methodology & Limitations — ATO Detection

This note documents the methodology decisions and the honest limitations of the
ATO sub-topic. It is written to be dropped almost directly into the group report
(Methodology / Discussion / Limitations sections).

---

## 1. The data leakage problem (and the fix)

**What went wrong in the first version.** The original notebooks reported an
LSTM ROC-AUC of **0.944**. Two issues inflated that number:

1. **Label leakage.** The label is defined as
   `ato_proxy = isFraud AND signal_count >= 2`. The original feature set
   included `signal_count` and `time_since_prev` (and `signal_velocity` is
   literally `time_since_prev < 7200`). So the model was given columns that
   directly define the answer — it could partly "read" the label.
2. **Split leakage.** Sequences were split randomly over rows. Because several
   sequences come from the same card, near-identical sequences ended up in both
   the train and test sets.

**The fix (in `src/`).**

- Every label-defining column is now declared in `config.LABEL_ONLY_COLUMNS` and
  removed before training. The model only sees genuine, non-label features.
- The train/test split is done **by card (`uid`)**, not by row, using a
  stratified split so attacks appear on both sides. One card can never be in both
  sets.
- The scaler and the imputation medians are fit on **training rows only**, so no
  test information leaks into preprocessing.

**Result.** On the clean, leak-free data (real, from `evaluation_v2.json`):
the **LSTM scores ROC-AUC 0.905 / PR-AUC 0.0275**, the Random Forest 0.917 /
0.0269, and Isolation Forest 0.863 / 0.0183. The high ROC-AUC with a tiny PR-AUC
is exactly what you expect for a 0.4%-positive problem, which is why we report
PR-AUC alongside ROC-AUC. Interestingly, the leaked notebook LSTM (0.944) and the
clean LSTM (0.905) are not far apart on ROC-AUC — but the clean score is the
trustworthy one, and the leak removal mainly shows up in the (still very low)
precision once you pick an operating threshold. This is a useful research
finding: it shows how easily proxy-label pipelines can leak, and that ROC-AUC can
stay high while the practically-relevant precision stays poor.

---

## 2. Honesty about the features ("behavioural biometrics")

The proposal slides describe behavioural signals such as *typing cadence,
navigation-path entropy, device fingerprint and login time-of-day*. **The
IEEE-CIS dataset does not contain these.** It is a transaction dataset, not a
session/keystroke dataset.

What the model actually uses are transaction-level and identity-level proxies:

- `TransactionAmt`, deviation of the amount from the card's mean
- a rough account-age proxy (`account_age_dt`) and hour-of-day
- a rolling transaction count
- device type / device-info frequency encoding
- the numeric `id_*` identity columns

The real SHAP feature ranking confirms this proxy nature (top 6, from
`shap_feature_importance_v2.json`): **id_17, DeviceType_enc, txn_count_rolling,
DeviceInfo_enc, card6_enc, id_02**. The device-related fields dominate, which is
consistent with ATO (an attacker logs in from new hardware) — but they are
device/identity proxies, not keystroke/biometric signals. Note the old leaked
SHAP list had `signal_count` near the top; that was itself a leak and is gone now.

These are **proxies** for behaviour, not true behavioural biometrics. The report
should state this clearly: the project demonstrates the *approach* (sequence
modelling of a user's recent activity for ATO), using the closest signals the
public dataset allows. Real deployment would need genuine session/biometric
telemetry.

---

## 3. Validity of the proxy label

There is no ground-truth ATO label in IEEE-CIS, so we built a proxy
(fraud + at least 2 of 3 behavioural signals). This is a standard approach when
sub-category labels are missing, but it has limits:

- The label can only ever be a subset of `isFraud`, so it inherits any labelling
  noise in the original fraud flag.
- The "2 of 3 signals" rule is a design choice. A sensitivity check (e.g.
  requiring 1 vs 2 vs 3 signals) should be reported so readers can see how much
  the label definition changes the positive count and the results.
- Because the label is derived from the same data the model sees, some residual
  association between features and label is unavoidable. We removed the *direct*
  leaks (Section 1); the remaining association is an inherent limitation of
  proxy labelling and bounds how high a fair score can go.

---

## 4. The recall / precision trade-off

ATO is extremely rare (~0.40% of rows; 442 positives in the 118,963-sequence test
set), so a single threshold cannot give both high recall and usable precision.
Real **LSTM** operating points (from `evaluation_v2.json`):

| Operating point | Threshold | Recall | Precision |
|---|---|---|---|
| Best F1 | 0.95 | 0.07 | 0.049 |
| Catch ~90% of attacks | 0.45 | 0.90 | 0.019 |

Reaching the proposal's 90% recall target collapses precision to ~2% — the model
flags a large number of sessions to catch most attacks, which would flood
analysts. So the proposal target (recall > 0.90 at useful precision) is **not
met** on a fair evaluation — and the report should say so. (The Random Forest
behaves the same way: at ~90% recall its precision is ~2% too.)

**Suggested operating policy (instead of one threshold):** a two-tier rule that
matches the proposal's "step-up authentication" idea:

- **High score → block** the session outright.
- **Medium score → step-up authentication** (ask for MFA / extra verification).
- **Low score → allow.**

This keeps friction low for most users while still challenging the riskiest
sessions, and it is the realistic way to use a low-precision, rare-event model.

---

## 5. Model comparison (honest take)

On the clean, leak-free data the **LSTM and the Random Forest are essentially
tied**: LSTM ROC-AUC 0.905 / PR-AUC 0.0275 vs RF 0.917 / 0.0269. The LSTM is
marginally ahead on PR-AUC (the metric that matters most here) but marginally
behind on ROC-AUC — so neither model clearly wins. Likely reasons:

- Most cards have very few transactions (median 2), so the 10-step sequences are
  mostly zero-padding and the LSTM has little temporal pattern to learn.
- A flattened Random Forest captures cross-feature interactions well on tabular
  data.
- The LSTM is trained with undersampling and a modest epoch budget; more tuning
  could help.

That the LSTM does not beat a simple baseline does not invalidate the sub-topic —
it is an honest, evidence-based outcome: on this dataset, for this proxy task, the
sequence approach is limited by short per-card histories. The contribution is the
*method* (leak-free sequence modelling of recent activity for ATO, with an
explicit recall/precision policy), not a claim that LSTM is the single best model.

---

## 6. Ethical and regulatory notes (unchanged, still valid)

- **GDPR Article 22 / FCA Consumer Duty** require explainable automated
  decisions. SHAP feature importance provides the audit trail.
- A **human-review tier** (the "step-up authentication" middle band) keeps a
  human in the loop, as Article 22 expects.
- The dataset is anonymised and public, used within the Kaggle licence; no PII is
  retained.
