"""
Central settings for the ATO detection pipeline.

Everything that other modules need to agree on (file paths, the random seed,
the sliding-window size, and which columns are allowed to be model inputs)
lives here. Keeping it in one place makes the whole project reproducible:
change a value once and every step uses the new value.
"""

from pathlib import Path

# --- Project folders -------------------------------------------------------
# PROJECT_ROOT is the ato_detection folder (two levels up from this file:
# src/config.py -> src -> ato_detection).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Make sure the output folders exist so saving never fails.
for _folder in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# --- Reproducibility -------------------------------------------------------
# One seed used everywhere (numpy, sklearn, tensorflow) so results repeat.
RANDOM_SEED = 42

# --- Source data -----------------------------------------------------------
# df_ato_labeled.parquet is the richest merged file: it already has `uid`,
# `prev_device` and the full id_* columns. We rebuild the labels from it.
SOURCE_PARQUET = PROCESSED_DIR / "df_ato_labeled.parquet"

# --- Sequence model settings ----------------------------------------------
# The LSTM looks at the last WINDOW_SIZE transactions of a card as one sequence.
WINDOW_SIZE = 10

# Share of cards (uids) held out for the test set. We split by card, not by
# row, so the same card can never appear in both train and test.
TEST_SIZE = 0.20

# --- Leakage control -------------------------------------------------------
# These columns are used ONLY to BUILD the ato_proxy label. They must never be
# given to the model as inputs, otherwise the model would "see the answer".
#   signal_count   -> the label is literally `isFraud AND signal_count >= 2`
#   time_since_prev-> signal_velocity is `time_since_prev < 7200`
#   device_changed / is_proxy / signal_* -> direct parts of the label
LABEL_ONLY_COLUMNS = [
    "signal_device",
    "signal_session",
    "signal_velocity",
    "signal_count",
    "device_changed",
    "is_proxy",
    "time_since_prev",
    "prev_device",
    "prev_dt",
]

# Columns that are identifiers or targets, never features.
ID_AND_TARGET_COLUMNS = ["TransactionID", "uid", "isFraud", "ato_proxy"]
