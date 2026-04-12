from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING

from . import _cli_common, daily_taq_paths
from .write_filtered_quote_parquet import write_filtered_quote_parquets

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)
TARGET_DATES: tuple[str, ...] = ("20251031", "20251103")
DEFAULT_QUOTE_COLUMNS: tuple[str, ...] = ("Exchange", "Participant_Timestamp")
SYMBOL_PREFIX = ""


def build_quote_summary(
    *,
    target_dates: Sequence[str] = TARGET_DATES,
    symbol_prefix: str = SYMBOL_PREFIX,
    columns: Sequence[str] = DEFAULT_QUOTE_COLUMNS,
) -> dict[str, dict[str, object]]:
    """Write filtered quote parquet files for the configured dates.

    Parameters
    ----------
    target_dates
        Trading dates formatted as ``YYYYMMDD``.
    symbol_prefix
        Prefix used to select symbols from the Daily TAQ master file.
    columns
        Raw quote columns to retain in the output parquet files.

    Returns
    -------
    dict[str, dict[str, object]]
        Per-date summary including symbol count, audit path, and written
        parquet paths.

    """
    summary: dict[str, dict[str, object]] = {}
    for date_yyyymmdd, symbols in _cli_common.symbols_by_date(target_dates, symbol_prefix).items():
        logger.info(
            "quote CLI started date=%s symbol_prefix=%r",
            date_yyyymmdd,
            symbol_prefix,
        )
        written_paths = write_filtered_quote_parquets(
            symbols,
            date_yyyymmdd,
            columns=columns,
        )
        summary[date_yyyymmdd] = _cli_common.build_cli_date_summary(
            symbol_prefix=symbol_prefix,
            symbol_count=len(symbols),
            columns=columns,
            audit_path=daily_taq_paths.quote_audit_path(date_yyyymmdd),
            paths=written_paths,
        )
        logger.info(
            "quote CLI completed date=%s symbol_count=%d written=%d",
            date_yyyymmdd,
            len(symbols),
            len(written_paths),
        )
    return summary


def main() -> int:
    """Run the quote parquet-writing CLI.

    Returns
    -------
    int
        Process exit code.

    """
    _cli_common.configure_cli_logging()
    logger.info("quote CLI batch started target_dates=%s", list(TARGET_DATES))
    summary = build_quote_summary()
    logger.info("quote CLI batch completed dates=%d", len(summary))
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
