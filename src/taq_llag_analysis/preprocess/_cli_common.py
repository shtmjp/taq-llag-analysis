from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import inventory

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def configure_cli_logging() -> None:
    """Configure INFO-level logging for preprocess CLI entrypoints."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def symbols_by_date(
    target_dates: Sequence[str],
    symbol_prefix: str,
) -> dict[str, list[str]]:
    """Resolve requested symbols for each target date.

    Parameters
    ----------
    target_dates
        Trading dates formatted as ``YYYYMMDD``.
    symbol_prefix
        Prefix used to select symbols from the Daily TAQ master file.

    Returns
    -------
    dict[str, list[str]]
        Symbols keyed by trading date.

    """
    return {
        date_yyyymmdd: inventory.symbols_with_prefix(date_yyyymmdd, symbol_prefix)
        for date_yyyymmdd in target_dates
    }


def build_cli_date_summary(
    *,
    symbol_prefix: str,
    symbol_count: int,
    columns: Sequence[str],
    audit_path: Path,
    paths: Mapping[str, Path],
) -> dict[str, object]:
    """Build one CLI summary payload for a single dataset/date.

    Parameters
    ----------
    symbol_prefix
        Prefix used to select symbols from the Daily TAQ master file.
    symbol_count
        Number of requested symbols for the date.
    columns
        Output columns retained in the parquet files.
    audit_path
        Audit JSON path for the date.
    paths
        Written parquet paths keyed by symbol.

    Returns
    -------
    dict[str, object]
        JSON-serializable summary payload.

    """
    return {
        "symbol_prefix": symbol_prefix,
        "symbol_count": symbol_count,
        "columns": list(columns),
        "audit_path": str(audit_path),
        "paths": {symbol: str(path) for symbol, path in paths.items()},
    }
