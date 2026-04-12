from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from . import audit, daily_taq_paths, quote_logic, quote_pipeline, runtime

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


def _quote_shards(symbols: Sequence[str]) -> list[str]:
    return sorted({symbol[0].upper() for symbol in symbols})


def _write_quote_shard_parquets(
    *,
    shard: str,
    input_path: Path,
    shard_symbols: Sequence[str],
    date_yyyymmdd: str,
    requested_columns: Sequence[str] | None,
    output_columns: Sequence[str],
) -> tuple[dict[str, Path], int, float, float]:
    include_symbol = "Symbol" in output_columns
    selected_lf = quote_pipeline.selected_quote_lazy_frame(
        input_path,
        symbols=shard_symbols,
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

    # Quote symbols are shard-local, so each shard can be written directly to
    # final per-symbol parquet paths without cross-shard merges.
    def output_path_provider(args: pl.io.partition.FileProviderArgs) -> Path:
        symbol = str(args.partition_keys.item(0, "Symbol"))
        output_path = daily_taq_paths.quote_output_path(symbol, date_yyyymmdd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return Path(symbol) / f"quote_{date_yyyymmdd}.parquet"

    selected_lf.sink_parquet(
        pl.PartitionBy(
            daily_taq_paths.QUOTE_OUTPUT_DIR,
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
        for symbol in shard_symbols
        if (output_path := daily_taq_paths.quote_output_path(symbol, date_yyyymmdd)).exists()
    }
    logger.info(
        "quote shard processed date=%s shard=%s requested_symbols=%d "
        "written_symbols=%d n_filtered_rows_total=%d "
        "collect_elapsed_sec=%.3f write_elapsed_sec=%.3f",
        date_yyyymmdd,
        shard,
        len(shard_symbols),
        len(written_paths),
        n_filtered_rows_total,
        collect_elapsed_sec,
        write_elapsed_sec,
    )
    return written_paths, n_filtered_rows_total, collect_elapsed_sec, write_elapsed_sec


def write_filtered_quote_parquets(
    symbols: Sequence[str],
    date_yyyymmdd: str,
    *,
    columns: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write filtered Daily TAQ quote parquet files for one trading date.

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
        quote row.

    """
    requested_symbols = list(symbols)
    output_columns = quote_logic.output_columns(columns)
    audit_path = daily_taq_paths.quote_audit_path(date_yyyymmdd)

    start_time = time.perf_counter()
    shards = _quote_shards(requested_symbols) if requested_symbols else []
    input_paths = daily_taq_paths.quote_input_paths(date_yyyymmdd, shards)
    logger.info(
        "quote filtering started date=%s requested_symbols=%d "
        "requested_columns=%s input_file_count=%d",
        date_yyyymmdd,
        len(requested_symbols),
        list(columns) if columns is not None else None,
        len(input_paths),
    )

    if not requested_symbols:
        payload = audit.build_filter_audit_payload(
            dataset="quote",
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
        daily_taq_paths.quote_output_path(symbol, date_yyyymmdd).unlink(missing_ok=True)

    symbols_by_shard = {
        shard: [symbol for symbol in requested_symbols if symbol[0].upper() == shard]
        for shard in shards
    }
    written_paths: dict[str, Path] = {}
    n_filtered_rows_total = 0
    collect_elapsed_sec = 0.0
    write_elapsed_sec = 0.0
    for shard, input_path in zip(shards, input_paths, strict=True):
        (
            shard_written_paths,
            shard_n_filtered_rows_total,
            shard_collect_elapsed_sec,
            shard_write_elapsed_sec,
        ) = _write_quote_shard_parquets(
            shard=shard,
            input_path=input_path,
            shard_symbols=symbols_by_shard[shard],
            date_yyyymmdd=date_yyyymmdd,
            requested_columns=columns,
            output_columns=output_columns,
        )
        written_paths.update(shard_written_paths)
        n_filtered_rows_total += shard_n_filtered_rows_total
        collect_elapsed_sec += shard_collect_elapsed_sec
        write_elapsed_sec += shard_write_elapsed_sec
    logger.info(
        "quote filtering collected date=%s n_filtered_rows_total=%d elapsed_sec=%.3f",
        date_yyyymmdd,
        n_filtered_rows_total,
        collect_elapsed_sec,
    )
    total_elapsed_sec = runtime.elapsed_seconds(start_time)

    logger.info(
        "quote filtering wrote date=%s written_symbols=%d "
        "write_elapsed_sec=%.3f total_elapsed_sec=%.3f",
        date_yyyymmdd,
        len(written_paths),
        write_elapsed_sec,
        total_elapsed_sec,
    )

    written_symbols = [symbol for symbol in requested_symbols if symbol in written_paths]
    payload = audit.build_filter_audit_payload(
        dataset="quote",
        date_yyyymmdd=date_yyyymmdd,
        input_paths=input_paths,
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
