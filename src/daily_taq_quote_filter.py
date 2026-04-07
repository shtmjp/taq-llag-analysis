from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


DEFAULT_QUOTE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "Time",
    "Participant_Timestamp",
    "Exchange",
    "Symbol",
    "Bid_Price",
    "Bid_Size",
    "Offer_Price",
    "Offer_Size",
    "Quote_Condition",
    "Quote_Cancel_Correction",
)

DEFAULT_QUOTE_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "Time": pl.Int64,
    "Participant_Timestamp": pl.Int64,
    "Exchange": pl.String,
    "Symbol": pl.String,
    "Bid_Price": pl.Float64,
    "Bid_Size": pl.Int64,
    "Offer_Price": pl.Float64,
    "Offer_Size": pl.Int64,
    "Quote_Condition": pl.String,
    "Quote_Cancel_Correction": pl.String,
}


def time_to_seconds_expr(time_col: str = "Time") -> pl.Expr:
    """Convert a Daily TAQ quote time to integer seconds.

    Parameters
    ----------
    time_col
        Column name containing the Daily TAQ quote time encoded as
        ``HHMMSSxxxxxxxxx``.

    Returns
    -------
    polars.Expr
        Expression evaluating to seconds since midnight.

    """
    time_value = pl.col(time_col)
    hh = time_value // 10_000_000_000_000
    mm = (time_value // 100_000_000_000) % 100
    ss = (time_value // 1_000_000_000) % 100
    return hh * 3600 + mm * 60 + ss


def filtered_quote_lazy_frame(
    input_path: Path,
    *,
    symbols: Sequence[str] | None = None,
    exchanges: Sequence[str] | None = None,
    use_market_hours: bool = True,
    market_open_sec: int = 35_100,
    market_close_sec: int = 56_700,
    all_columns: bool = True,
    add_time_sec: bool = False,
) -> pl.LazyFrame:
    """Build a filtered LazyFrame for Daily TAQ quote data from a BBO shard.

    Parameters
    ----------
    input_path
        Path to ``SPLITS_US_ALL_BBO_*`` data for one date and shard.
    symbols
        Optional symbol whitelist.
    exchanges
        Optional exchange whitelist.
    use_market_hours
        If True, keep only quotes inside the inclusive market-hours window.
    market_open_sec
        Inclusive market-open time in seconds since midnight.
    market_close_sec
        Inclusive market-close time in seconds since midnight.
    all_columns
        If True, keep all input columns.
    add_time_sec
        If True, include a derived ``time_sec`` column in the output.

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
        schema_overrides=DEFAULT_QUOTE_SCHEMA_OVERRIDES,
    )

    if symbols:
        lf = lf.filter(pl.col("Symbol").is_in(symbols))
    if exchanges:
        lf = lf.filter(pl.col("Exchange").is_in(exchanges))

    lf = lf.filter(
        (pl.col("Bid_Price") > 0)
        & (pl.col("Offer_Price") > 0)
        & (pl.col("Bid_Price") < pl.col("Offer_Price")),
    )

    time_sec = time_to_seconds_expr()
    if use_market_hours:
        lf = lf.filter((time_sec >= market_open_sec) & (time_sec <= market_close_sec))

    if add_time_sec:
        lf = lf.with_columns(time_sec.alias("time_sec"))

    if not all_columns:
        output_columns = list(DEFAULT_QUOTE_OUTPUT_COLUMNS)
        if add_time_sec:
            output_columns.append("time_sec")
        lf = lf.select(output_columns)

    return lf
