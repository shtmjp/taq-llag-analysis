from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import polars as pl

from . import _daily_taq as daily_taq

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


logger = logging.getLogger(__name__)


def _filtered_trade_lazy_frame(
    input_path: Path,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    market_open_sec: int,
    market_close_sec: int,
) -> pl.LazyFrame:
    lf = pl.scan_csv(
        input_path,
        separator="|",
        has_header=True,
        comment_prefix="END",
        schema_overrides=daily_taq.TRADE_SCHEMA_OVERRIDES,
    )

    if symbols:
        lf = lf.filter(pl.col("Symbol").is_in(symbols))
    if exchanges:
        lf = lf.filter(pl.col("Exchange").is_in(exchanges))

    participant_timestamp_sec = daily_taq.hms_integer_seconds_expr("Participant Timestamp")
    return lf.filter(
        (participant_timestamp_sec >= market_open_sec)
        & (participant_timestamp_sec <= market_close_sec),
    )


def write_filtered_trade_parquets(
    symbols: Sequence[str],
    date_yyyymmdd: str,
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write filtered Daily TAQ trade parquet files for one trading date.

    Parameters
    ----------
    symbols
        Symbols to keep.
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    columns
        Optional raw-column subset to keep in the output parquet files. When
        omitted, all raw columns are written.

    Returns
    -------
    dict[str, pathlib.Path]
        Output parquet path for each symbol that has at least one filtered
        trade row.

    """
    output_columns = daily_taq.resolve_output_columns(columns, daily_taq.TRADE_RAW_COLUMNS)
    collect_columns = daily_taq.append_if_missing(output_columns, "Symbol")
    input_path = daily_taq.trade_input_path(date_yyyymmdd)
    audit_path = daily_taq.trade_audit_path(date_yyyymmdd)
    requested_symbols = list(symbols)

    start_time = time.perf_counter()
    logger.info(
        "trade filtering started date=%s requested_symbols=%d "
        "requested_columns=%s input_file_count=%d",
        date_yyyymmdd,
        len(requested_symbols),
        list(columns) if columns is not None else None,
        int(bool(requested_symbols)),
    )

    if not requested_symbols:
        payload = {
            "created_at_utc": daily_taq.utc_now_timestamp(),
            "dataset": "trade",
            "date_yyyymmdd": date_yyyymmdd,
            "input_paths": [],
            "requested_symbols": requested_symbols,
            "requested_columns": list(columns) if columns is not None else None,
            "resolved_output_columns": output_columns,
            "written_symbols": [],
            "n_requested_symbols": 0,
            "n_written_symbols": 0,
            "n_filtered_rows_total": 0,
            "stage_elapsed_sec": {
                "collect": 0.0,
                "write": 0.0,
                "total": time.perf_counter() - start_time,
            },
        }
        daily_taq.write_audit_json(audit_path, payload)
        return {}

    written_paths, n_filtered_rows_total, collect_elapsed_sec, write_elapsed_sec = (
        daily_taq.stream_lazy_frames_to_symbol_parquets(
            [
                (
                    "trade",
                    _filtered_trade_lazy_frame(
                        input_path,
                        symbols=requested_symbols,
                        exchanges=None,
                        market_open_sec=daily_taq.MARKET_OPEN_SEC,
                        market_close_sec=daily_taq.MARKET_CLOSE_SEC,
                    ).select(collect_columns),
                ),
            ],
            output_columns=output_columns,
            output_path_for_symbol=lambda symbol: (
                daily_taq.TRADE_OUTPUT_DIR / symbol / f"trade_{date_yyyymmdd}.parquet"
            ),
        )
    )
    logger.info(
        "trade filtering collected date=%s n_filtered_rows_total=%d elapsed_sec=%.3f",
        date_yyyymmdd,
        n_filtered_rows_total,
        collect_elapsed_sec,
    )
    total_elapsed_sec = time.perf_counter() - start_time

    logger.info(
        "trade filtering wrote date=%s written_symbols=%d "
        "write_elapsed_sec=%.3f total_elapsed_sec=%.3f",
        date_yyyymmdd,
        len(written_paths),
        write_elapsed_sec,
        total_elapsed_sec,
    )

    payload = {
        "created_at_utc": daily_taq.utc_now_timestamp(),
        "dataset": "trade",
        "date_yyyymmdd": date_yyyymmdd,
        "input_paths": [str(input_path)],
        "requested_symbols": requested_symbols,
        "requested_columns": list(columns) if columns is not None else None,
        "resolved_output_columns": output_columns,
        "written_symbols": [symbol for symbol in requested_symbols if symbol in written_paths],
        "n_requested_symbols": len(requested_symbols),
        "n_written_symbols": len(written_paths),
        "n_filtered_rows_total": n_filtered_rows_total,
        "stage_elapsed_sec": {
            "collect": collect_elapsed_sec,
            "write": write_elapsed_sec,
            "total": total_elapsed_sec,
        },
    }
    daily_taq.write_audit_json(audit_path, payload)
    return {
        symbol: written_paths[symbol] for symbol in requested_symbols if symbol in written_paths
    }
