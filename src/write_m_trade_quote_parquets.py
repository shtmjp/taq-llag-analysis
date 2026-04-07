from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from write_filtered_quote_parquet import write_filtered_quote_parquets
from write_filtered_trade_parquet import write_filtered_trade_parquets

MASTER_INPUT_DIR = Path("data/dailyTAQ/MASTER")
TARGET_DATES: tuple[str, ...] = ("20251031", "20251103")
SYMBOL_PREFIX = "M"


def master_input_path(date_yyyymmdd: str) -> Path:
    """Resolve the Daily TAQ master file path for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.

    Returns
    -------
    pathlib.Path
        Plain-text master path when present, otherwise the `.gz` path.

    """
    plain_path = MASTER_INPUT_DIR / f"EQY_US_ALL_REF_MASTER_{date_yyyymmdd}"
    gz_path = plain_path.with_suffix(".gz")
    return plain_path if plain_path.exists() else gz_path


def symbols_with_prefix(date_yyyymmdd: str, symbol_prefix: str) -> list[str]:
    """Load symbols whose ticker starts with the requested prefix.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    symbol_prefix
        Prefix to match against the `Symbol` column.

    Returns
    -------
    list[str]
        Sorted symbol list for the requested date.

    """
    return (
        pl.read_csv(
            master_input_path(date_yyyymmdd),
            separator="|",
            has_header=True,
            comment_prefix="END",
            encoding="latin1",
        )
        .filter(pl.col("Symbol").str.starts_with(symbol_prefix))
        .select("Symbol")
        .unique()
        .sort("Symbol")
        .get_column("Symbol")
        .to_list()
    )


def build_summary() -> dict[str, dict[str, object]]:
    """Write trade and quote parquet files for the configured prefix and dates.

    Returns
    -------
    dict[str, dict[str, object]]
        Per-date summary including symbol count and written parquet paths.

    """
    quote_shard = SYMBOL_PREFIX[0]
    summary: dict[str, dict[str, object]] = {}

    for date_yyyymmdd in TARGET_DATES:
        symbols = symbols_with_prefix(date_yyyymmdd, SYMBOL_PREFIX)
        trade_paths = write_filtered_trade_parquets(symbols, date_yyyymmdd)
        quote_paths = write_filtered_quote_parquets(symbols, date_yyyymmdd, quote_shard)
        summary[date_yyyymmdd] = {
            "symbol_prefix": SYMBOL_PREFIX,
            "symbol_count": len(symbols),
            "quote_shard": quote_shard,
            "trade_paths": {symbol: str(path) for symbol, path in trade_paths.items()},
            "quote_paths": {symbol: str(path) for symbol, path in quote_paths.items()},
        }

    return summary


def main() -> int:
    """Run the parquet-writing workflow for `M`-prefixed symbols.

    Returns
    -------
    int
        Process exit code.

    """
    summary = build_summary()
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
