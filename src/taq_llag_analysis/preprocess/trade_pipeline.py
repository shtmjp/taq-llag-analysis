from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from . import schemas, trade_logic

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def scan_trade_lazy_frame(input_path: Path) -> pl.LazyFrame:
    """Scan one raw Daily TAQ trade file into a lazy frame."""
    return pl.scan_csv(
        input_path,
        separator="|",
        has_header=True,
        comment_prefix="END",
        schema_overrides=schemas.TRADE_SCHEMA_OVERRIDES,
    )


def partition_columns(requested_columns: Sequence[str] | None) -> tuple[str, ...]:
    """Return the selected trade columns including the internal partition key."""
    output_columns = trade_logic.output_columns(requested_columns)
    if "Symbol" in output_columns:
        return output_columns
    return (*output_columns, "Symbol")


def selected_trade_lazy_frame(
    input_path: Path,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    columns: Sequence[str] | None,
) -> pl.LazyFrame:
    """Build the filtered trade lazy frame ready for partitioned writing.

    Parameters
    ----------
    input_path
        Raw trade input path.
    symbols
        Optional symbols to retain.
    exchanges
        Optional exchanges to retain.
    columns
        Final trade parquet columns requested by the caller.

    Returns
    -------
    polars.LazyFrame
        Filtered and selected lazy frame, including ``Symbol`` when needed
        for partitioned writing.

    """
    return trade_logic.apply_trade_filters(
        scan_trade_lazy_frame(input_path),
        symbols=symbols,
        exchanges=exchanges,
    ).select(partition_columns(columns))
