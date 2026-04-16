from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from . import audit, daily_taq_paths, runtime, trade_logic, trade_pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


def _write_trade_parquets(
    *,
    input_path: Path,
    requested_symbols: Sequence[str],
    date_yyyymmdd: str,
    requested_columns: Sequence[str] | None,
    output_columns: Sequence[str],
) -> tuple[dict[str, Path], int, float, float]:
    include_symbol = "Symbol" in output_columns
    selected_lf = trade_pipeline.selected_trade_lazy_frame(
        input_path,
        symbols=requested_symbols,
        exchanges=None,
        columns=requested_columns,
    )

    count_start = time.perf_counter()
    n_filtered_rows_total = int(
        selected_lf.select(pl.len().alias("n_rows")).collect(engine="streaming").item(0, "n_rows"),
    )
    collect_elapsed_sec = runtime.elapsed_seconds(count_start)
    if n_filtered_rows_total == 0:
        return {}, 0, collect_elapsed_sec, 0.0

    write_start = time.perf_counter()

    def output_path_provider(args: pl.io.partition.FileProviderArgs) -> Path:
        symbol = str(args.partition_keys.item(0, "Symbol"))
        output_path = daily_taq_paths.trade_output_path(symbol, date_yyyymmdd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return Path(symbol) / f"trade_{date_yyyymmdd}.parquet"

    selected_lf.sink_parquet(
        pl.PartitionBy(
            daily_taq_paths.TRADE_OUTPUT_DIR,
            key="Symbol",
            include_key=include_symbol,
            file_path_provider=output_path_provider,
            max_rows_per_file=None,
            approximate_bytes_per_file=None,
        ),
        mkdir=True,
        statistics=False,
        engine="streaming",
    )
    write_elapsed_sec = runtime.elapsed_seconds(write_start)
    written_paths = {
        symbol: output_path
        for symbol in requested_symbols
        if (output_path := daily_taq_paths.trade_output_path(symbol, date_yyyymmdd)).exists()
    }
    return written_paths, n_filtered_rows_total, collect_elapsed_sec, write_elapsed_sec


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
    output_columns = trade_logic.output_columns(columns)
    input_path = daily_taq_paths.trade_input_path(date_yyyymmdd)
    audit_path = daily_taq_paths.trade_audit_path(date_yyyymmdd)
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
        payload = audit.build_filter_audit_payload(
            dataset="trade",
            date_yyyymmdd=date_yyyymmdd,
            input_paths=(),
            requested_symbols=requested_symbols,
            requested_columns=columns,
            resolved_output_columns=output_columns,
            written_symbols=(),
            n_filtered_rows_total=0,
            collect_elapsed_sec=0.0,
            write_elapsed_sec=0.0,
            total_elapsed_sec=runtime.elapsed_seconds(start_time),
        )
        audit.write_audit_json(audit_path, payload)
        return {}

    for symbol in requested_symbols:
        daily_taq_paths.trade_output_path(symbol, date_yyyymmdd).unlink(missing_ok=True)

    written_paths, n_filtered_rows_total, collect_elapsed_sec, write_elapsed_sec = (
        _write_trade_parquets(
            input_path=input_path,
            requested_symbols=requested_symbols,
            date_yyyymmdd=date_yyyymmdd,
            requested_columns=columns,
            output_columns=output_columns,
        )
    )
    logger.info(
        "trade filtering collected date=%s n_filtered_rows_total=%d elapsed_sec=%.3f",
        date_yyyymmdd,
        n_filtered_rows_total,
        collect_elapsed_sec,
    )
    total_elapsed_sec = runtime.elapsed_seconds(start_time)

    logger.info(
        "trade filtering wrote date=%s written_symbols=%d "
        "write_elapsed_sec=%.3f total_elapsed_sec=%.3f",
        date_yyyymmdd,
        len(written_paths),
        write_elapsed_sec,
        total_elapsed_sec,
    )

    written_symbols = [symbol for symbol in requested_symbols if symbol in written_paths]
    payload = audit.build_filter_audit_payload(
        dataset="trade",
        date_yyyymmdd=date_yyyymmdd,
        input_paths=(input_path,),
        requested_symbols=requested_symbols,
        requested_columns=columns,
        resolved_output_columns=output_columns,
        written_symbols=written_symbols,
        n_filtered_rows_total=n_filtered_rows_total,
        collect_elapsed_sec=collect_elapsed_sec,
        write_elapsed_sec=write_elapsed_sec,
        total_elapsed_sec=total_elapsed_sec,
    )
    audit.write_audit_json(audit_path, payload)
    return {
        symbol: written_paths[symbol] for symbol in requested_symbols if symbol in written_paths
    }
