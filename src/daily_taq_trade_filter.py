from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


DEFAULT_TRADE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "Exchange",
    "Symbol",
    "Trade Price",
    "Trade Volume",
    "Sale Condition",
    "Participant Timestamp",
    "Trade Correction Indicator",
    "Trade Stop Stock Indicator",
)

DEFAULT_TRADE_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "Time": pl.Int64,
    "Exchange": pl.String,
    "Symbol": pl.String,
    "Sale Condition": pl.String,
    "Trade Volume": pl.Int64,
    "Trade Price": pl.Float64,
    "Trade Stop Stock Indicator": pl.String,
    "Trade Correction Indicator": pl.Int64,
    "Sequence Number": pl.Int64,
    "Trade Id": pl.Int64,
    "Source of Trade": pl.String,
    "Trade Reporting Facility": pl.String,
    "Participant Timestamp": pl.Int64,
    "Trade Reporting Facility TRF Timestamp": pl.String,
    "Trade Through Exempt Indicator": pl.Int64,
}


def participant_timestamp_sec_expr(
    timestamp_col: str = "Participant Timestamp",
) -> pl.Expr:
    """Convert a Daily TAQ participant timestamp to integer seconds.

    Parameters
    ----------
    timestamp_col
        Column name containing the Daily TAQ participant timestamp encoded as
        ``HHMMSSxxxxxxxxx``.

    Returns
    -------
    polars.Expr
        Expression evaluating to seconds since midnight.

    """
    timestamp = pl.col(timestamp_col)
    hh = timestamp // 10_000_000_000_000
    mm = (timestamp // 100_000_000_000) % 100
    ss = (timestamp // 1_000_000_000) % 100
    return hh * 3600 + mm * 60 + ss


def participant_timestamp_time_expr(
    timestamp_col: str = "Participant Timestamp",
) -> pl.Expr:
    """Convert a Daily TAQ participant timestamp to floating-point seconds.

    Parameters
    ----------
    timestamp_col
        Column name containing the Daily TAQ participant timestamp encoded as
        ``HHMMSSxxxxxxxxx``.

    Returns
    -------
    polars.Expr
        Expression evaluating to seconds since midnight, including the
        sub-second component.

    """
    timestamp = pl.col(timestamp_col)
    hh = (timestamp // 10_000_000_000_000).cast(pl.Float64)
    mm = ((timestamp // 100_000_000_000) % 100).cast(pl.Float64)
    ss = ((timestamp // 1_000_000_000) % 100).cast(pl.Float64)
    subsec = (timestamp % 1_000_000_000).cast(pl.Float64) / 1_000_000_000
    return hh * 3600 + mm * 60 + ss + subsec


def filtered_trade_lazy_frame(
    input_path: Path,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    market_open_sec: int,
    market_close_sec: int,
    all_columns: bool,
    add_time_columns: bool,
) -> pl.LazyFrame:
    """Build a filtered LazyFrame for Daily TAQ trade data.

    Parameters
    ----------
    input_path
        Path to `EQY_US_ALL_TRADE_*` (plain text or `.gz`).
    symbols
        Optional symbol whitelist.
    exchanges
        Optional exchange whitelist.
    market_open_sec
        Inclusive market-open time in seconds since midnight.
    market_close_sec
        Inclusive market-close time in seconds since midnight.
    all_columns
        If True, keep all input columns.
    add_time_columns
        If True, include integer-second and floating-point time columns in the
        output.

    Returns
    -------
    polars.LazyFrame
        Filtered lazy query.

    """
    lf = pl.scan_csv(
        input_path,
        separator="|",
        has_header=True,
        comment_prefix="END",
        schema_overrides=DEFAULT_TRADE_SCHEMA_OVERRIDES,
    )

    if symbols:
        lf = lf.filter(pl.col("Symbol").is_in(symbols))
    if exchanges:
        lf = lf.filter(pl.col("Exchange").is_in(exchanges))

    participant_timestamp_sec = participant_timestamp_sec_expr()
    lf = lf.filter(
        (participant_timestamp_sec >= market_open_sec)
        & (participant_timestamp_sec <= market_close_sec),
    )

    if add_time_columns:
        lf = lf.with_columns(
            participant_timestamp_sec.alias("participant_timestamp_sec"),
            participant_timestamp_time_expr().alias("participant_timestamp_time"),
        )

    if not all_columns:
        output_columns = list(DEFAULT_TRADE_OUTPUT_COLUMNS)
        if add_time_columns:
            output_columns.extend(
                ["participant_timestamp_sec", "participant_timestamp_time"],
            )
        lf = lf.select(output_columns)

    return lf
