"""Tests for src/features/build.py -- categorical encoding, behaviour
features, leak-free feature selection, and sliding-window sequence
construction."""

import numpy as np
import pandas as pd

from src import config
from src.features.build import (
    _build_sequences,
    _choose_feature_columns,
    _clean_numeric,
    _encode_categoricals,
    _engineer_behaviour,
)


def test_encode_categoricals_device_type_mapping():
    df = pd.DataFrame({
        "DeviceType": ["mobile", "desktop", None],
        "DeviceInfo": ["iPhone", "iPhone", "Windows"],
        "card4": ["visa", "mastercard", "visa"],
        "card6": ["debit", "credit", "debit"],
    })
    out = _encode_categoricals(df)
    assert list(out["DeviceType_enc"]) == [1, 2, 0]
    # DeviceInfo encoded by frequency: iPhone appears twice, Windows once
    assert out.loc[0, "DeviceInfo_enc"] == 2
    assert out.loc[2, "DeviceInfo_enc"] == 1
    assert out["card4_enc"].notna().all()


def test_engineer_behaviour_txn_count_and_amt_deviation():
    df = pd.DataFrame({
        "uid": ["A", "A", "A"],
        "TransactionDT": [0, 100, 200],
        "TransactionAmt": [10.0, 10.0, 100.0],
    })
    out = _engineer_behaviour(df)
    assert list(out["txn_count_rolling"]) == [1, 2, 3]
    # the big outlier (100) should have a much larger |amt_deviation| than the two 10s
    assert abs(out.loc[2, "amt_deviation"]) > abs(out.loc[0, "amt_deviation"])
    assert (out["account_age_dt"] == out["TransactionDT"] - 0).all()


def test_choose_feature_columns_excludes_label_only_and_id_columns():
    df = pd.DataFrame({
        "TransactionAmt": [1.0], "amt_deviation": [0.0], "account_age_dt": [0.0],
        "hour_of_day": [0.0], "txn_count_rolling": [1], "DeviceType_enc": [1],
        "DeviceInfo_enc": [1], "card4_enc": [0], "card6_enc": [0],
        "id_02": [1.0],
        # label-only / id columns that must NEVER be selected as features
        "signal_count": [2], "device_changed": [1], "time_since_prev": [10.0],
        "TransactionID": [1], "uid": ["A"], "isFraud": [0], "ato_proxy": [0],
    })
    cols = _choose_feature_columns(df)
    banned = set(config.LABEL_ONLY_COLUMNS) | set(config.ID_AND_TARGET_COLUMNS)
    assert banned.isdisjoint(set(cols))
    assert "id_02" in cols
    assert "TransactionAmt" in cols


def test_clean_numeric_clips_extreme_outliers():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1_000_000.0]})
    out = _clean_numeric(df.copy(), ["x"])
    assert out["x"].max() < 1_000_000.0


def test_clean_numeric_removes_infinities():
    df = pd.DataFrame({"x": [1.0, np.inf, -np.inf, 2.0]})
    out = _clean_numeric(df.copy(), ["x"])
    assert not np.isinf(out["x"]).any()


def test_build_sequences_pads_short_histories_at_the_front():
    feature_cols = ["f1"]
    df = pd.DataFrame({
        "uid": ["A", "A"],
        "TransactionDT": [0, 100],
        "TransactionID": [1, 2],
        "ato_proxy": [0, 1],
        "f1": [5.0, 9.0],
    })
    X, y, owner_uid, txn_ids = _build_sequences(df, feature_cols)
    # only 1 sample: uid A's 2nd transaction (index 0 has no history to predict from)
    assert X.shape == (1, config.WINDOW_SIZE, 1)
    assert y[0] == 1
    assert txn_ids[0] == 2
    assert owner_uid[0] == "A"
    # front should be zero-padded, only the last row holds real history (f1=5.0)
    assert X[0, -1, 0] == 5.0
    assert (X[0, :-1, 0] == 0.0).all()


def test_build_sequences_skips_cards_with_a_single_transaction():
    feature_cols = ["f1"]
    df = pd.DataFrame({
        "uid": ["A"], "TransactionDT": [0], "TransactionID": [1],
        "ato_proxy": [0], "f1": [1.0],
    })
    X, y, owner_uid, txn_ids = _build_sequences(df, feature_cols)
    assert X.shape[0] == 0
