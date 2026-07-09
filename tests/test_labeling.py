"""Tests for src/data/labeling.py -- the three behavioural signals and the
final ato_proxy rule (fraud AND at least 2 of 3 signals)."""

import pandas as pd

from src.data.labeling import build_labels


def _base_df(rows: list[dict]) -> pd.DataFrame:
    """Fill in the columns build_labels()/its helpers require, so each test
    only has to specify what actually varies."""
    defaults = {
        "DeviceInfo": "Windows",
        "id_31": "chrome 70.0",
        "id_01": 0.0, "id_02": 0.0, "id_03": 0.0, "id_04": 0.0, "id_05": 0.0,
        "id_06": 0.0, "id_07": 0.0, "id_08": 0.0, "id_09": 0.0,
        "id_10": 0.0, "id_11": 0.0,
        "isFraud": 0,
    }
    out = []
    for i, row in enumerate(rows):
        merged = {**defaults, **row}
        merged.setdefault("TransactionID", i)
        out.append(merged)
    return pd.DataFrame(out)


def test_signal_device_fires_on_device_change():
    df = _base_df([
        {"uid": "A", "TransactionDT": 0, "DeviceInfo": "Windows"},
        {"uid": "A", "TransactionDT": 100_000, "DeviceInfo": "iPhone"},  # far apart, device changed
    ])
    result = build_labels(df)
    # second row: device changed from the first -> signal_device fires
    assert result.iloc[1]["signal_device"] == 1
    assert result.iloc[0]["signal_device"] == 0  # no previous device to compare to


def test_signal_velocity_fires_on_fast_transactions():
    df = _base_df([
        {"uid": "A", "TransactionDT": 0},
        {"uid": "A", "TransactionDT": 3600},  # 1 hour later -> under the 2h threshold
    ])
    result = build_labels(df)
    assert result.iloc[1]["signal_velocity"] == 1
    assert result.iloc[0]["signal_velocity"] == 0  # no previous transaction to compare to


def test_signal_velocity_does_not_fire_when_spaced_out():
    df = _base_df([
        {"uid": "A", "TransactionDT": 0},
        {"uid": "A", "TransactionDT": 100_000},  # >2h later
    ])
    result = build_labels(df)
    assert result.iloc[1]["signal_velocity"] == 0


def test_signal_session_fires_on_outlier_id_value():
    # uid with a stable id_01 history, then one wildly different value
    rows = [{"uid": "A", "TransactionDT": i * 100_000, "id_01": 1.0} for i in range(5)]
    rows.append({"uid": "A", "TransactionDT": 500_000, "id_01": 500.0})
    df = _base_df(rows)
    result = build_labels(df)
    assert result.iloc[-1]["signal_session"] == 1


def test_ato_proxy_requires_fraud_and_two_signals():
    """Two signals fire (device change + velocity) but isFraud=0 -> no ato_proxy."""
    df = _base_df([
        {"uid": "A", "TransactionDT": 0, "DeviceInfo": "Windows", "isFraud": 0},
        {"uid": "A", "TransactionDT": 3600, "DeviceInfo": "iPhone", "isFraud": 0},
    ])
    result = build_labels(df)
    assert result.iloc[1]["signal_count"] >= 2
    assert result.iloc[1]["ato_proxy"] == 0


def test_ato_proxy_fires_when_fraud_and_two_signals():
    df = _base_df([
        {"uid": "A", "TransactionDT": 0, "DeviceInfo": "Windows", "isFraud": 1},
        {"uid": "A", "TransactionDT": 3600, "DeviceInfo": "iPhone", "isFraud": 1},
    ])
    result = build_labels(df)
    assert result.iloc[1]["signal_count"] >= 2
    assert result.iloc[1]["ato_proxy"] == 1


def test_ato_proxy_does_not_fire_with_only_one_signal():
    """Fraud + device change only (velocity doesn't fire, far apart in time)
    -> signal_count == 1 -> ato_proxy stays 0."""
    df = _base_df([
        {"uid": "A", "TransactionDT": 0, "DeviceInfo": "Windows", "isFraud": 1},
        {"uid": "A", "TransactionDT": 200_000, "DeviceInfo": "iPhone", "isFraud": 1},
    ])
    result = build_labels(df)
    assert result.iloc[1]["signal_count"] == 1
    assert result.iloc[1]["ato_proxy"] == 0


def test_is_proxy_signal_detects_vpn_keyword():
    df = _base_df([
        {"uid": "A", "TransactionDT": 0, "id_31": "vpn service"},
    ])
    result = build_labels(df)
    assert result.iloc[0]["is_proxy"] == 1
    assert result.iloc[0]["signal_device"] == 1
