from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


RAW_ROOT_DIR = Path("data/datasets/dailytaq2")
TRADE_OUTPUT_DIR = Path("data/filtered/trade")
QUOTE_OUTPUT_DIR = Path("data/filtered/quote")
TRADE_AUDIT_DIR = TRADE_OUTPUT_DIR / "_audit"
QUOTE_AUDIT_DIR = QUOTE_OUTPUT_DIR / "_audit"


def master_input_path(date_yyyymmdd: str) -> Path:
    """Resolve the Daily TAQ master file path for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.

    Returns
    -------
    pathlib.Path
        Gzipped master-file path for the requested date.

    """
    year = date_yyyymmdd[:4]
    year_month = date_yyyymmdd[:6]
    return (
        RAW_ROOT_DIR
        / f"EQY_US_ALL_REF_MASTER_{year}"
        / f"EQY_US_ALL_REF_MASTER_{year_month}"
        / f"EQY_US_ALL_REF_MASTER_{date_yyyymmdd}.gz"
    )


def trade_input_path(date_yyyymmdd: str) -> Path:
    """Resolve the Daily TAQ trade file path for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.

    Returns
    -------
    pathlib.Path
        Gzipped trade-file path for the requested date.

    """
    year = date_yyyymmdd[:4]
    year_month = date_yyyymmdd[:6]
    return (
        RAW_ROOT_DIR
        / f"EQY_US_ALL_TRADE_{year}"
        / f"EQY_US_ALL_TRADE_{year_month}"
        / f"EQY_US_ALL_TRADE_{date_yyyymmdd}"
        / f"EQY_US_ALL_TRADE_{date_yyyymmdd}.gz"
    )


def quote_input_paths(date_yyyymmdd: str, shards: Sequence[str]) -> list[Path]:
    """Resolve Daily TAQ quote shard paths for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    shards
        Quote shard tokens such as ``"A"`` or ``"M"``.

    Returns
    -------
    list[pathlib.Path]
        Gzipped BBO shard paths for the requested date.

    """
    year = date_yyyymmdd[:4]
    year_month = date_yyyymmdd[:6]
    base_dir = (
        RAW_ROOT_DIR
        / f"SPLITS_US_ALL_BBO_{year}"
        / f"SPLITS_US_ALL_BBO_{year_month}"
        / f"SPLITS_US_ALL_BBO_{date_yyyymmdd}"
    )
    return [
        base_dir / f"SPLITS_US_ALL_BBO_{shard}_{date_yyyymmdd}.gz" for shard in sorted(set(shards))
    ]


def trade_output_path(symbol: str, date_yyyymmdd: str) -> Path:
    """Return the filtered trade parquet path for one symbol and date."""
    return TRADE_OUTPUT_DIR / symbol / f"trade_{date_yyyymmdd}.parquet"


def quote_output_path(symbol: str, date_yyyymmdd: str) -> Path:
    """Return the filtered quote parquet path for one symbol and date."""
    return QUOTE_OUTPUT_DIR / symbol / f"quote_{date_yyyymmdd}.parquet"


def trade_audit_path(date_yyyymmdd: str) -> Path:
    """Return the audit JSON path for one filtered trade date."""
    return TRADE_AUDIT_DIR / f"trade_{date_yyyymmdd}.json"


def quote_audit_path(date_yyyymmdd: str) -> Path:
    """Return the audit JSON path for one filtered quote date."""
    return QUOTE_AUDIT_DIR / f"quote_{date_yyyymmdd}.json"
