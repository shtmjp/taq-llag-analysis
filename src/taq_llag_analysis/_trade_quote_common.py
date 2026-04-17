from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


TRADE_BASE_DIR = Path("data/filtered/trade")
QUOTE_BASE_DIR = Path("data/filtered/quote")
MODE_SUMMARY_OUTPUT_BASE_DIR = Path("data/derived/trade_quote_modes")
CROSS_K_SUMMARY_FILENAME = "cross_k_summary.csv"
MODES_FILENAME = "modes.csv"
OBS_WINDOW = (35_100.0, 56_700.0)
KERNEL = "tent"
ALLOW_NOT_SIMPLE = True
CROSS_K_WINDOWS: dict[str, tuple[float, float]] = {
    "cross_k_neg_1e3_1e4": (-1e-3, -1e-4),
    "cross_k_neg_1e4_1e5": (-1e-4, -1e-5),
    "cross_k_neg_1e5_0": (-1e-5, 0.0),
    "cross_k_pos_0_1e5": (0.0, 1e-5),
    "cross_k_pos_1e5_1e4": (1e-5, 1e-4),
    "cross_k_pos_1e4_1e3": (1e-4, 1e-3),
}


def _timestamp_to_time_expr(timestamp_col: str) -> pl.Expr:
    timestamp = pl.col(timestamp_col)
    # TAQ の HHMMSSnnnnnnnnn 形式を float 秒へ直して、比較と推定入力をそろえる。
    hh = (timestamp // 10_000_000_000_000).cast(pl.Float64)
    mm = ((timestamp // 100_000_000_000) % 100).cast(pl.Float64)
    ss = ((timestamp // 1_000_000_000) % 100).cast(pl.Float64)
    subsec = (timestamp % 1_000_000_000).cast(pl.Float64) / 1_000_000_000.0
    return (hh * 3600.0 + mm * 60.0 + ss + subsec).alias("event_time")


def _event_time_frame(
    path: Path,
    timestamp_col: str,
    exchanges: Iterable[str] | None,
    *,
    obs_window: tuple[float, float],
) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema={"Exchange": pl.String, "event_time": pl.Float64})

    event_df = pl.read_parquet(path, columns=["Exchange", timestamp_col])
    if exchanges is not None:
        selected_exchanges = sorted(set(exchanges))
        if not selected_exchanges:
            return pl.DataFrame(schema={"Exchange": pl.String, "event_time": pl.Float64})
        event_df = event_df.filter(pl.col("Exchange").is_in(selected_exchanges))

    return (
        event_df.unique(subset=["Exchange", timestamp_col], keep="any")
        .with_columns(_timestamp_to_time_expr(timestamp_col))
        .filter(pl.col("event_time").is_between(obs_window[0], obs_window[1]))
        .select(["Exchange", "event_time"])
        .sort(["Exchange", "event_time"])
    )


def _scan_event_counts(
    path: Path,
    timestamp_col: str,
    *,
    obs_window: tuple[float, float],
) -> dict[str, int]:
    counts_df = (
        _event_time_frame(
            path,
            timestamp_col,
            exchanges=None,
            obs_window=obs_window,
        )
        .group_by("Exchange")
        .agg(pl.len().alias("n_events"))
        .sort("Exchange")
    )
    return {exchange: int(n_events) for exchange, n_events in counts_df.iter_rows()}


def _event_arrays_by_exchange(
    path: Path,
    timestamp_col: str,
    exchanges: Iterable[str],
    *,
    obs_window: tuple[float, float],
) -> dict[str, np.ndarray]:
    selected_exchanges = sorted(set(exchanges))
    if not selected_exchanges:
        return {}

    event_df = _event_time_frame(
        path,
        timestamp_col,
        exchanges=selected_exchanges,
        obs_window=obs_window,
    )

    arrays_by_exchange: dict[str, np.ndarray] = {}
    for exchange_df in event_df.partition_by("Exchange", maintain_order=False):
        exchange = exchange_df.get_column("Exchange")[0]
        arrays_by_exchange[exchange] = np.asarray(
            exchange_df.get_column("event_time").to_numpy(),
            dtype=np.float64,
        )
    return arrays_by_exchange
