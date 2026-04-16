from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from . import schemas

if TYPE_CHECKING:
    from collections.abc import Sequence


MARKET_OPEN_SEC = 35_100
MARKET_CLOSE_SEC = 56_700


def filter_required_columns() -> tuple[str, ...]:
    """Return the raw columns required to evaluate trade filters."""
    return ("Symbol", "Exchange", "Participant Timestamp")


def output_columns(requested_columns: Sequence[str] | None) -> tuple[str, ...]:
    """Resolve the final trade parquet columns.

    Parameters
    ----------
    requested_columns
        Optional caller-specified output columns.

    Returns
    -------
    tuple[str, ...]
        Final parquet columns in output order.

    """
    if requested_columns is None:
        return schemas.TRADE_RAW_COLUMNS
    return tuple(requested_columns)


def filter_conditions(
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    market_open_sec: int = MARKET_OPEN_SEC,
    market_close_sec: int = MARKET_CLOSE_SEC,
) -> tuple[tuple[str, pl.Expr], ...]:
    """Return the ordered trade filter conditions for one run.

    Parameters
    ----------
    symbols
        Optional symbols to retain.
    exchanges
        Optional exchanges to retain.
    market_open_sec
        Inclusive market-open bound in seconds from midnight.
    market_close_sec
        Inclusive market-close bound in seconds from midnight.

    Returns
    -------
    tuple[tuple[str, polars.Expr], ...]
        Ordered named filter conditions.

    """
    conditions: list[tuple[str, pl.Expr]] = []
    if symbols:
        conditions.append(("symbol", pl.col("Symbol").is_in(symbols)))
    if exchanges:
        conditions.append(("exchange", pl.col("Exchange").is_in(exchanges)))
    participant_timestamp_sec = _hms_integer_seconds_expr("Participant Timestamp")
    conditions.append(
        (
            "market_hours",
            (participant_timestamp_sec >= market_open_sec)
            & (participant_timestamp_sec <= market_close_sec),
        ),
    )
    return tuple(conditions)


def apply_trade_filters(
    lazy_frame: pl.LazyFrame,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    market_open_sec: int = MARKET_OPEN_SEC,
    market_close_sec: int = MARKET_CLOSE_SEC,
) -> pl.LazyFrame:
    """Apply the ordered trade filters to a lazy frame.

    Parameters
    ----------
    lazy_frame
        Unfiltered trade lazy frame.
    symbols
        Optional symbols to retain.
    exchanges
        Optional exchanges to retain.
    market_open_sec
        Inclusive market-open bound in seconds from midnight.
    market_close_sec
        Inclusive market-close bound in seconds from midnight.

    Returns
    -------
    polars.LazyFrame
        Filtered trade lazy frame.

    """
    filtered_lf = lazy_frame
    for _, condition in filter_conditions(
        symbols=symbols,
        exchanges=exchanges,
        market_open_sec=market_open_sec,
        market_close_sec=market_close_sec,
    ):
        filtered_lf = filtered_lf.filter(condition)
    return filtered_lf


def _hms_integer_seconds_expr(time_col: str) -> pl.Expr:
    time_value = pl.col(time_col)
    hh = time_value // 10_000_000_000_000
    mm = (time_value // 100_000_000_000) % 100
    ss = (time_value // 1_000_000_000) % 100
    return hh * 3600 + mm * 60 + ss
