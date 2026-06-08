"""
Build the ATO proxy label.

The IEEE-CIS dataset has no real "account takeover" label, only a general
`isFraud` flag. Following Alghofaili et al. (2020), we mark a transaction as
account-takeover (ato_proxy = 1) when it is fraud AND it also shows at least
two of three behavioural warning signs.

IMPORTANT: the signals built here are used ONLY to make the label. They are
listed in config.LABEL_ONLY_COLUMNS and are removed before training, so the
model never sees them. This is what stops the model from "cheating".
"""

import numpy as np
import pandas as pd

from src import config


def _signal_device(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 1: the device/identity changed, or a proxy/VPN was used.

    A real attacker often logs in from new hardware after stealing a password,
    or hides behind an anonymity network.
    """
    # Look at the previous device used on the same card.
    df["prev_device"] = df.groupby("uid")["DeviceInfo"].shift(1)

    # Device changed between two transactions of the same card.
    df["device_changed"] = (
        df["DeviceInfo"].notna()
        & df["prev_device"].notna()
        & (df["DeviceInfo"] != df["prev_device"])
    ).astype(int)

    # id_31 sometimes names the network; flag proxy / anonymous / VPN.
    df["is_proxy"] = (
        df["id_31"].astype(str).str.contains("proxy|anonymous|vpn", case=False, na=False)
    ).astype(int)

    df["signal_device"] = ((df["device_changed"] == 1) | (df["is_proxy"] == 1)).astype(int)
    return df


def _signal_session(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 2: the session behaviour is unusual for this card.

    For each card we check the id_01..id_11 columns (login / session fields).
    If any value is more than 2 standard deviations away from that card's own
    history, the session looks abnormal.

    This is written with groupby.transform (vectorised) instead of a Python
    loop over cards, so it runs in seconds instead of many minutes. A column
    only counts if the card has at least 3 values for it and a non-zero spread.
    """
    id_cols = [f"id_0{i}" for i in range(1, 10)] + ["id_10", "id_11"]
    id_cols = [c for c in id_cols if c in df.columns]

    anomaly = pd.Series(False, index=df.index)
    for col in id_cols:
        # Force numeric; non-numeric id columns cannot give a z-score.
        values = pd.to_numeric(df[col], errors="coerce")
        grp = values.groupby(df["uid"])
        mu = grp.transform("mean")
        sigma = grp.transform("std")
        count = grp.transform("count")
        z = (values - mu) / sigma
        col_anom = (z.abs() > 2.0) & (count >= 3) & (sigma > 0)
        anomaly = anomaly | col_anom.fillna(False)

    df["signal_session"] = anomaly.astype(int)
    return df


def _signal_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 3: transactions come too fast (scripted / automated behaviour).

    Two or more transactions on the same card within 2 hours is unusual for a
    normal user.
    """
    df["prev_dt"] = df.groupby("uid")["TransactionDT"].shift(1)
    df["time_since_prev"] = df["TransactionDT"] - df["prev_dt"]
    df["signal_velocity"] = (
        df["time_since_prev"].notna() & (df["time_since_prev"] < 7200)
    ).astype(int)
    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the three signals and the final ato_proxy label to the dataframe."""
    # Process each card in time order so "previous transaction" makes sense.
    df = df.sort_values(["uid", "TransactionDT"]).reset_index(drop=True)

    df = _signal_device(df)
    df = _signal_session(df)
    df = _signal_velocity(df)

    # Count how many of the three signals fired.
    df["signal_count"] = df["signal_device"] + df["signal_session"] + df["signal_velocity"]

    # Final rule: fraud AND at least two signals -> treat as account takeover.
    df["ato_proxy"] = ((df["isFraud"] == 1) & (df["signal_count"] >= 2)).astype(int)
    return df


def run() -> pd.DataFrame:
    """Load the merged data, build labels, save and return the result."""
    print(f"[labeling] reading {config.SOURCE_PARQUET.name} ...")
    df = pd.read_parquet(config.SOURCE_PARQUET)

    df = build_labels(df)

    out_path = config.PROCESSED_DIR / "df_labeled_v2.parquet"
    df.to_parquet(out_path, index=False)

    pos = int(df["ato_proxy"].sum())
    print(f"[labeling] rows={len(df):,}  ato_proxy=1: {pos:,} "
          f"({df['ato_proxy'].mean() * 100:.3f}%)")
    print(f"[labeling] saved -> {out_path}")
    return df


if __name__ == "__main__":
    run()
