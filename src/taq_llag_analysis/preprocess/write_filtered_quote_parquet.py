from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from . import _daily_taq as daily_taq

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = logging.getLogger(__name__)


def _filtered_quote_lazy_frame(
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
        schema_overrides=daily_taq.QUOTE_SCHEMA_OVERRIDES,
    )

    if symbols:
        lf = lf.filter(pl.col("Symbol").is_in(symbols))
    if exchanges:
        lf = lf.filter(pl.col("Exchange").is_in(exchanges))

    time_sec = daily_taq.hms_integer_seconds_expr("Time")
    return lf.filter(
        (pl.col("Bid_Price") > 0)
        & (pl.col("Offer_Price") > 0)
        & (pl.col("Bid_Price") < pl.col("Offer_Price"))
        & (time_sec >= market_open_sec)
        & (time_sec <= market_close_sec),
    )


def _quote_shards(symbols: Sequence[str]) -> list[str]:
    return sorted({symbol[0].upper() for symbol in symbols})


def _quote_output_path(symbol: str, date_yyyymmdd: str) -> Path:
    return daily_taq.QUOTE_OUTPUT_DIR / symbol / f"quote_{date_yyyymmdd}.parquet"


def _write_quote_shard_parquets(
    *,
    shard: str,
    input_path: Path,
    shard_symbols: Sequence[str],
    date_yyyymmdd: str,
    output_columns: Sequence[str],
) -> tuple[dict[str, Path], int, float, float]:
    partition_columns = daily_taq.append_if_missing(output_columns, "Symbol")
    include_symbol = "Symbol" in output_columns
    selected_lf = _filtered_quote_lazy_frame(
        input_path,
        symbols=shard_symbols,
        exchanges=None,
        market_open_sec=daily_taq.MARKET_OPEN_SEC,
        market_close_sec=daily_taq.MARKET_CLOSE_SEC,
    ).select(partition_columns)

    count_start = time.perf_counter()
    n_filtered_rows_total = int(
        selected_lf.select(pl.len().alias("n_rows")).collect(engine="streaming").item(0, "n_rows"),
    )
    collect_elapsed_sec = time.perf_counter() - count_start

    if n_filtered_rows_total == 0:
        return {}, 0, collect_elapsed_sec, 0.0

    write_start = time.perf_counter()

    # Quote symbols are shard-local, so each shard can be written directly to
    # final per-symbol parquet paths without cross-shard merges.
    def output_path_provider(args: pl.io.partition.FileProviderArgs) -> Path:
        symbol = str(args.partition_keys.item(0, "Symbol"))
        output_path = _quote_output_path(symbol, date_yyyymmdd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return Path(symbol) / f"quote_{date_yyyymmdd}.parquet"

    selected_lf.sink_parquet(
        pl.PartitionBy(
            daily_taq.QUOTE_OUTPUT_DIR,
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
    write_elapsed_sec = time.perf_counter() - write_start

    written_paths = {
        symbol: output_path
        for symbol in shard_symbols
        if (output_path := _quote_output_path(symbol, date_yyyymmdd)).exists()
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
    output_columns = daily_taq.resolve_output_columns(columns, daily_taq.QUOTE_RAW_COLUMNS)
    audit_path = daily_taq.quote_audit_path(date_yyyymmdd)

    start_time = time.perf_counter()
    shards = _quote_shards(requested_symbols) if requested_symbols else []
    input_paths = daily_taq.quote_input_paths(date_yyyymmdd, shards)
    logger.info(
        "quote filtering started date=%s requested_symbols=%d "
        "requested_columns=%s input_file_count=%d",
        date_yyyymmdd,
        len(requested_symbols),
        list(columns) if columns is not None else None,
        len(input_paths),
    )

    if not requested_symbols:
        payload = {
            "created_at_utc": daily_taq.utc_now_timestamp(),
            "dataset": "quote",
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

    for symbol in requested_symbols:
        _quote_output_path(symbol, date_yyyymmdd).unlink(missing_ok=True)

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
    total_elapsed_sec = time.perf_counter() - start_time

    logger.info(
        "quote filtering wrote date=%s written_symbols=%d "
        "write_elapsed_sec=%.3f total_elapsed_sec=%.3f",
        date_yyyymmdd,
        len(written_paths),
        write_elapsed_sec,
        total_elapsed_sec,
    )

    payload = {
        "created_at_utc": daily_taq.utc_now_timestamp(),
        "dataset": "quote",
        "date_yyyymmdd": date_yyyymmdd,
        "input_paths": [str(path) for path in input_paths],
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
