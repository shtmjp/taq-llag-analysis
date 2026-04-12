from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

from . import _daily_taq as daily_taq
from .write_filtered_quote_parquet import write_filtered_quote_parquets
from .write_filtered_trade_parquet import write_filtered_trade_parquets

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)
TARGET_DATES: tuple[str, ...] = ("20251031", "20251103")
DEFAULT_TRADE_COLUMNS: tuple[str, ...] = ("Exchange", "Participant Timestamp")
DEFAULT_QUOTE_COLUMNS: tuple[str, ...] = ("Exchange", "Participant_Timestamp")
SYMBOL_PREFIX = ""


def build_summary(
    *,
    target_dates: Sequence[str] = TARGET_DATES,
    symbol_prefix: str = SYMBOL_PREFIX,
    trade_columns: Sequence[str] = DEFAULT_TRADE_COLUMNS,
    quote_columns: Sequence[str] = DEFAULT_QUOTE_COLUMNS,
) -> dict[str, dict[str, object]]:
    """Write filtered trade and quote parquet files for the configured dates.

    Parameters
    ----------
    target_dates
        Trading dates formatted as ``YYYYMMDD``.
    symbol_prefix
        Prefix used to select symbols from the Daily TAQ master file.
    trade_columns
        Raw trade columns to retain in the output parquet files.
    quote_columns
        Raw quote columns to retain in the output parquet files.

    Returns
    -------
    dict[str, dict[str, object]]
        Per-date summary including symbol count, audit paths, and written
        parquet paths.

    """
    summary: dict[str, dict[str, object]] = {}

    for date_yyyymmdd in target_dates:
        logger.info(
            "batch filtering started date=%s symbol_prefix=%r",
            date_yyyymmdd,
            symbol_prefix,
        )
        symbols = daily_taq.symbols_with_prefix(date_yyyymmdd, symbol_prefix)
        trade_paths = write_filtered_trade_parquets(
            symbols,
            date_yyyymmdd,
            columns=trade_columns,
        )
        quote_paths = write_filtered_quote_parquets(
            symbols,
            date_yyyymmdd,
            columns=quote_columns,
        )
        summary[date_yyyymmdd] = {
            "symbol_prefix": symbol_prefix,
            "symbol_count": len(symbols),
            "trade_columns": list(trade_columns),
            "quote_columns": list(quote_columns),
            "trade_audit_path": str(daily_taq.trade_audit_path(date_yyyymmdd)),
            "quote_audit_path": str(daily_taq.quote_audit_path(date_yyyymmdd)),
            "trade_paths": {symbol: str(path) for symbol, path in trade_paths.items()},
            "quote_paths": {symbol: str(path) for symbol, path in quote_paths.items()},
        }
        logger.info(
            "batch filtering completed date=%s symbol_count=%d trade_written=%d quote_written=%d",
            date_yyyymmdd,
            len(symbols),
            len(trade_paths),
            len(quote_paths),
        )

    return summary


def build_all_summary(
    *,
    target_dates: Sequence[str] = TARGET_DATES,
    trade_columns: Sequence[str] = DEFAULT_TRADE_COLUMNS,
    quote_columns: Sequence[str] = DEFAULT_QUOTE_COLUMNS,
) -> dict[str, dict[str, object]]:
    """Write filtered trade and quote parquet files for all symbols.

    Parameters
    ----------
    target_dates
        Trading dates formatted as ``YYYYMMDD``.
    trade_columns
        Raw trade columns to retain in the output parquet files.
    quote_columns
        Raw quote columns to retain in the output parquet files.

    Returns
    -------
    dict[str, dict[str, object]]
        Per-date summary including symbol count, audit paths, and written
        parquet paths.

    """
    return build_summary(
        target_dates=target_dates,
        symbol_prefix=SYMBOL_PREFIX,
        trade_columns=trade_columns,
        quote_columns=quote_columns,
    )


def main() -> int:
    """Run the all-symbol parquet-writing workflow.

    Returns
    -------
    int
        Process exit code.

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("all-symbol batch started target_dates=%s", list(TARGET_DATES))
    summary = build_all_summary()
    logger.info("all-symbol batch completed dates=%d", len(summary))
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
