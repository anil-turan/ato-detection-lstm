"""
Turn the labeled transactions into model-ready sliding-window sequences.

Steps:
  1. Encode text columns (device, card type) into numbers.
  2. Build a few simple per-card behaviour features.
  3. Choose a LEAK-FREE feature list (drop every label-only column).
  4. Split the CARDS (uids) into train/test, so one card is never in both.
  5. Fit the scaler on TRAIN ONLY, then scale everything.
  6. Build sequences: each example = a card's last 10 transactions,
     and its label = the ato_proxy of the most recent transaction.

The split-by-card and train-only scaling are what keep the test score honest
(no information from the test set leaks into training).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to numbers (the LSTM needs all-numeric input)."""
    # Device type: mobile=1, desktop=2, missing=0.
    df["DeviceType_enc"] = df["DeviceType"].map({"mobile": 1, "desktop": 2}).fillna(0).astype(int)

    # Device info has too many values, so encode each by how often it appears.
    device_freq = df["DeviceInfo"].value_counts()
    df["DeviceInfo_enc"] = df["DeviceInfo"].map(device_freq).fillna(0).astype(int)

    # Card brand / type -> simple integer codes.
    for col in ("card4", "card6"):
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype("category").cat.codes
    return df


def _engineer_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """Add behaviour features that describe a card's own history.

    None of these are used to build the label, so they are safe features.
    """
    df = df.sort_values(["uid", "TransactionDT"]).reset_index(drop=True)

    # How many transactions this card has had so far.
    df["txn_count_rolling"] = df.groupby("uid").cumcount() + 1

    # How far this amount is from the card's usual amount.
    card_mean = df.groupby("uid")["TransactionAmt"].transform("mean")
    card_std = df.groupby("uid")["TransactionAmt"].transform("std").fillna(1.0)
    df["amt_deviation"] = (df["TransactionAmt"] - card_mean) / card_std

    # Rough "account age": seconds since the card's first transaction.
    card_first = df.groupby("uid")["TransactionDT"].transform("min")
    df["account_age_dt"] = df["TransactionDT"] - card_first

    # Hour of day (0-24), derived from the dataset's seconds clock.
    df["hour_of_day"] = (df["TransactionDT"] % 86400) / 3600
    return df


def _choose_feature_columns(df: pd.DataFrame) -> list:
    """Pick the leak-free numeric feature columns.

    We keep behaviour features, encoded categoricals and the numeric id_*
    columns. We DROP every label-only column and every id/target column.
    """
    banned = set(config.LABEL_ONLY_COLUMNS) | set(config.ID_AND_TARGET_COLUMNS)

    wanted = [
        "TransactionAmt", "amt_deviation", "account_age_dt",
        "hour_of_day", "txn_count_rolling",
        "DeviceType_enc", "DeviceInfo_enc", "card4_enc", "card6_enc",
    ]
    # Add numeric id_* columns (these are identity fields, fair to use).
    id_cols = [c for c in df.columns if c.startswith("id_")]

    candidates = wanted + id_cols
    feature_cols = []
    for c in candidates:
        if c in banned or c not in df.columns:
            continue
        # Keep only columns we can turn into numbers.
        as_num = pd.to_numeric(df[c], errors="coerce")
        if as_num.notna().any():
            feature_cols.append(c)
    return feature_cols


def _clean_numeric(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Make every feature numeric and remove infinities / extreme outliers."""
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # Clip very extreme values so scaling is stable.
    for c in feature_cols:
        lo, hi = df[c].quantile(0.001), df[c].quantile(0.999)
        df[c] = df[c].clip(lo, hi)
    return df


def _build_sequences(df: pd.DataFrame, feature_cols: list):
    """Make sliding-window sequences, keeping each card on one side of the split.

    Returns X (samples, WINDOW_SIZE, n_features), y (samples,), the matching
    uid for every sample, and the TransactionID of the latest step in each
    sequence. The TransactionID is what lets a downstream consumer join
    these risk scores onto their own outputs.
    """
    window = config.WINDOW_SIZE
    n_feat = len(feature_cols)

    seqs, labels, owner_uid, txn_ids = [], [], [], []
    for uid, group in df.groupby("uid"):
        group = group.sort_values("TransactionDT")
        feats = group[feature_cols].to_numpy(dtype="float32")
        ys = group["ato_proxy"].to_numpy()
        tids = group["TransactionID"].to_numpy()
        if len(group) < 2:
            continue  # a single transaction has no history to learn from
        for i in range(1, len(group)):
            start = max(0, i - window)
            seq = feats[start:i]
            if len(seq) < window:  # pad the front with zeros if too short
                pad = np.zeros((window - len(seq), n_feat), dtype="float32")
                seq = np.vstack([pad, seq])
            seqs.append(seq)
            labels.append(ys[i])
            owner_uid.append(uid)
            txn_ids.append(tids[i])  # id of the transaction being scored

    X = np.asarray(seqs, dtype="float32")
    y = np.asarray(labels, dtype="int8")
    owner_uid = np.asarray(owner_uid)
    txn_ids = np.asarray(txn_ids, dtype="int64")
    return X, y, owner_uid, txn_ids


def run() -> dict:
    """Run the whole feature build and save train/test arrays to disk."""
    np.random.seed(config.RANDOM_SEED)

    labeled_path = config.PROCESSED_DIR / "df_labeled_v2.parquet"
    print(f"[features] reading {labeled_path.name} ...")
    df = pd.read_parquet(labeled_path)

    df = _encode_categoricals(df)
    df = _engineer_behaviour(df)
    feature_cols = _choose_feature_columns(df)
    print(f"[features] using {len(feature_cols)} leak-free features")
    df = _clean_numeric(df, feature_cols)

    # --- Split CARDS into train/test (not rows) -----------------------------
    # Mark each card as positive if it has at least one ato_proxy=1 row, then
    # stratify the card split on that so both sides contain attacks.
    uid_has_pos = df.groupby("uid")["ato_proxy"].max()
    train_uids, test_uids = train_test_split(
        uid_has_pos.index.to_numpy(),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=uid_has_pos.to_numpy(),
    )
    train_mask = df["uid"].isin(set(train_uids))

    # --- Impute missing values with TRAIN medians only ----------------------
    medians = df.loc[train_mask, feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)
    medians.to_csv(config.OUTPUTS_DIR / "imputation_medians_v2.csv")

    # --- Scale using TRAIN rows only ---------------------------------------
    # Fit the scaler on training rows, then apply it to every row. This stops
    # test-set information from leaking into the scaling step.
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feature_cols])
    df[feature_cols] = scaler.transform(df[feature_cols])
    import pickle
    with open(config.OUTPUTS_DIR / "scaler_v2.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # --- Build sequences for train cards and test cards separately ----------
    print("[features] building sequences (this can take a minute) ...")
    df_train = df[df["uid"].isin(set(train_uids))]
    df_test = df[df["uid"].isin(set(test_uids))]
    X_train, y_train, _, ids_train = _build_sequences(df_train, feature_cols)
    X_test, y_test, _, ids_test = _build_sequences(df_test, feature_cols)

    np.save(config.PROCESSED_DIR / "X_train.npy", X_train)
    np.save(config.PROCESSED_DIR / "y_train.npy", y_train)
    np.save(config.PROCESSED_DIR / "X_test.npy", X_test)
    np.save(config.PROCESSED_DIR / "y_test.npy", y_test)
    # TransactionID per sequence, so downstream consumers can join on it.
    np.save(config.PROCESSED_DIR / "txn_ids_train.npy", ids_train)
    np.save(config.PROCESSED_DIR / "txn_ids_test.npy", ids_test)
    with open(config.PROCESSED_DIR / "feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    print(f"[features] train: {X_train.shape}  pos={int(y_train.sum()):,} "
          f"({y_train.mean()*100:.3f}%)")
    print(f"[features] test : {X_test.shape}  pos={int(y_test.sum()):,} "
          f"({y_test.mean()*100:.3f}%)")
    return {
        "feature_cols": feature_cols,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
    }


if __name__ == "__main__":
    run()
