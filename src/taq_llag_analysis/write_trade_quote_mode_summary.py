from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import ppllag
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ._trade_quote_common import (
    ALLOW_NOT_SIMPLE,
    CROSS_K_SUMMARY_FILENAME,
    CROSS_K_WINDOWS,
    KERNEL,
    MODE_SUMMARY_OUTPUT_BASE_DIR,
    MODES_FILENAME,
    OBS_WINDOW,
    QUOTE_BASE_DIR,
    TRADE_BASE_DIR,
    _event_arrays_by_exchange,
    _scan_event_counts,
)
from .preprocess.inventory import symbols_with_prefix

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


PairKey = tuple[str, str, str, str]
ScanKey = tuple[str, str]

TARGET_DATES: tuple[str, ...] = ("20251031", "20251103")
MIN_EVENTS_PER_SIDE = 100
N_JOBS = 1
U_RANGE = (-1.0, 1.0)
BW_CANDIDATES = np.array([1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
JOBLIB_BACKEND = "loky"

CROSS_K_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "date_yyyymmdd": pl.String,
    "symbol": pl.String,
    "trade_exchange": pl.String,
    "quote_exchange": pl.String,
    "status": pl.String,
    "n_trade_events": pl.Int64,
    "n_quote_events": pl.Int64,
    "n_shared_event_times": pl.Int64,
    "bandwidth": pl.Float64,
    "mode_count": pl.Int64,
    "closest_mode_to_zero_sec": pl.Float64,
    "elapsed_sec": pl.Float64,
    "cross_k_neg_1e3_1e4": pl.Float64,
    "cross_k_neg_1e4_1e5": pl.Float64,
    "cross_k_neg_1e5_0": pl.Float64,
    "cross_k_pos_0_1e5": pl.Float64,
    "cross_k_pos_1e5_1e4": pl.Float64,
    "cross_k_pos_1e4_1e3": pl.Float64,
    "error_type": pl.String,
    "error_message": pl.String,
}
CROSS_K_ONLY_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "date_yyyymmdd": pl.String,
    "symbol": pl.String,
    "trade_exchange": pl.String,
    "quote_exchange": pl.String,
    "status": pl.String,
    "n_trade_events": pl.Int64,
    "n_quote_events": pl.Int64,
    "n_shared_event_times": pl.Int64,
    "elapsed_sec": pl.Float64,
    "cross_k_neg_1e3_1e4": pl.Float64,
    "cross_k_neg_1e4_1e5": pl.Float64,
    "cross_k_neg_1e5_0": pl.Float64,
    "cross_k_pos_0_1e5": pl.Float64,
    "cross_k_pos_1e5_1e4": pl.Float64,
    "cross_k_pos_1e4_1e3": pl.Float64,
    "error_type": pl.String,
    "error_message": pl.String,
}
MODES_SCHEMA: dict[str, pl.DataType] = {
    "date_yyyymmdd": pl.String,
    "symbol": pl.String,
    "trade_exchange": pl.String,
    "quote_exchange": pl.String,
    "mode_index": pl.Int64,
    "mode_sec": pl.Float64,
}


@dataclass(frozen=True)
class _SymbolScan:
    date_yyyymmdd: str
    symbol: str
    trade_path: Path
    quote_path: Path
    trade_counts: dict[str, int]
    quote_counts: dict[str, int]

    @property
    def candidate_pairs(self) -> list[tuple[str, str]]:
        return _candidate_pairs(self.trade_counts, self.quote_counts)


@dataclass(frozen=True)
class _ScanJob:
    scan: _SymbolScan
    candidate_pairs: tuple[tuple[str, str], ...]

    @property
    def scan_key(self) -> ScanKey:
        return (self.scan.date_yyyymmdd, self.scan.symbol)

    @property
    def n_pairs(self) -> int:
        return len(self.candidate_pairs)


@dataclass(frozen=True)
class _ScanResult:
    scan_key: ScanKey
    cross_k_rows: list[dict[str, object]]
    mode_rows: list[dict[str, object]]

    @property
    def n_pairs(self) -> int:
        return len(self.cross_k_rows)


@dataclass(frozen=True)
class _ExistingOutputs:
    cross_k_df: pl.DataFrame
    modes_df: pl.DataFrame
    cross_k_rows_by_key: dict[PairKey, dict[str, object]]
    mode_counts_by_key: dict[PairKey, int]


def _csv_frame(
    rows: list[dict[str, object]],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(rows, schema=schema)


def _candidate_pairs(
    trade_counts: Mapping[str, int],
    quote_counts: Mapping[str, int],
) -> list[tuple[str, str]]:
    return [
        (trade_exchange, quote_exchange)
        for trade_exchange in sorted(trade_counts)
        for quote_exchange in sorted(quote_counts)
        if trade_exchange != quote_exchange
    ]


def _pair_key(
    date_yyyymmdd: str,
    symbol: str,
    trade_exchange: str,
    quote_exchange: str,
) -> PairKey:
    return (date_yyyymmdd, symbol, trade_exchange, quote_exchange)


def _scan_key_from_row(row: Mapping[str, object]) -> ScanKey:
    return (str(row["date_yyyymmdd"]), str(row["symbol"]))


def _cross_k_summary_path(output_dir: Path) -> Path:
    return output_dir / CROSS_K_SUMMARY_FILENAME


def _modes_path(output_dir: Path) -> Path:
    return output_dir / MODES_FILENAME


def _cross_k_summary_schema(*, cross_k_only: bool) -> Mapping[str, pl.DataType]:
    return CROSS_K_ONLY_SUMMARY_SCHEMA if cross_k_only else CROSS_K_SUMMARY_SCHEMA


def _read_csv_if_exists(
    path: Path,
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_csv(path, schema_overrides=schema)


def _write_csv_frame(path: Path, frame: pl.DataFrame) -> None:
    frame.write_csv(path)


def _append_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
    schema: Mapping[str, pl.DataType],
) -> None:
    if not rows:
        return
    frame = _csv_frame(rows, schema)
    include_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        frame.write_csv(handle, include_header=include_header)


def _csv_header(path: Path) -> tuple[str, ...] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
    return tuple(first_row) if first_row is not None else ()


def _ensure_compatible_output_dir(output_dir: Path, *, cross_k_only: bool) -> None:
    cross_k_summary_path = _cross_k_summary_path(output_dir)
    modes_path = _modes_path(output_dir)
    expected_cross_k_header = tuple(_cross_k_summary_schema(cross_k_only=cross_k_only))
    actual_cross_k_header = _csv_header(cross_k_summary_path)
    if actual_cross_k_header is not None and actual_cross_k_header != expected_cross_k_header:
        mode_name = "cross-K-only" if cross_k_only else "full"
        msg = (
            f"Output directory {output_dir} already contains {CROSS_K_SUMMARY_FILENAME} with "
            f"a schema incompatible with {mode_name} mode."
        )
        raise ValueError(msg)

    actual_modes_header = _csv_header(modes_path)
    if cross_k_only:
        if actual_modes_header is not None:
            msg = (
                f"Output directory {output_dir} cannot be reused with cross_k_only=True "
                f"because {MODES_FILENAME} already exists."
            )
            raise ValueError(msg)
        return

    if actual_cross_k_header is None and actual_modes_header is None:
        return
    if actual_cross_k_header is None or actual_modes_header is None:
        msg = (
            f"Output directory {output_dir} is incompatible with full mode because it does "
            f"not contain both {CROSS_K_SUMMARY_FILENAME} and {MODES_FILENAME}."
        )
        raise ValueError(msg)
    if actual_modes_header != tuple(MODES_SCHEMA):
        msg = (
            f"Output directory {output_dir} already contains {MODES_FILENAME} with an "
            "unexpected schema."
        )
        raise ValueError(msg)


def _ensure_output_files(output_dir: Path, *, cross_k_only: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_k_summary_path = _cross_k_summary_path(output_dir)
    if not cross_k_summary_path.exists():
        _write_csv_frame(
            cross_k_summary_path,
            pl.DataFrame(schema=_cross_k_summary_schema(cross_k_only=cross_k_only)),
        )
    if cross_k_only:
        return

    modes_path = _modes_path(output_dir)
    if not modes_path.exists():
        _write_csv_frame(modes_path, pl.DataFrame(schema=MODES_SCHEMA))


def _load_existing_outputs(output_dir: Path, *, cross_k_only: bool) -> _ExistingOutputs:
    cross_k_df = _read_csv_if_exists(
        _cross_k_summary_path(output_dir),
        _cross_k_summary_schema(cross_k_only=cross_k_only),
    )
    modes_df = (
        pl.DataFrame(schema=MODES_SCHEMA)
        if cross_k_only
        else _read_csv_if_exists(_modes_path(output_dir), MODES_SCHEMA)
    )

    cross_k_rows_by_key: dict[PairKey, dict[str, object]] = {}
    for row in cross_k_df.iter_rows(named=True):
        cross_k_rows_by_key[
            _pair_key(
                str(row["date_yyyymmdd"]),
                str(row["symbol"]),
                str(row["trade_exchange"]),
                str(row["quote_exchange"]),
            )
        ] = row

    mode_counts_by_key: dict[PairKey, int] = {}
    for row in modes_df.iter_rows(named=True):
        key = _pair_key(
            str(row["date_yyyymmdd"]),
            str(row["symbol"]),
            str(row["trade_exchange"]),
            str(row["quote_exchange"]),
        )
        mode_counts_by_key[key] = mode_counts_by_key.get(key, 0) + 1

    return _ExistingOutputs(
        cross_k_df=cross_k_df,
        modes_df=modes_df,
        cross_k_rows_by_key=cross_k_rows_by_key,
        mode_counts_by_key=mode_counts_by_key,
    )


def _drop_scan_rows(
    frame: pl.DataFrame,
    scan_keys: set[ScanKey],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    if not scan_keys or frame.is_empty():
        return frame
    kept_rows = [
        row for row in frame.iter_rows(named=True) if _scan_key_from_row(row) not in scan_keys
    ]
    return _csv_frame(kept_rows, schema)


def _cleanup_incomplete_scan_rows(
    output_dir: Path,
    existing_outputs: _ExistingOutputs,
    pending_scan_keys: set[ScanKey],
    *,
    cross_k_only: bool,
) -> None:
    if not pending_scan_keys:
        return
    cross_k_df = _drop_scan_rows(
        existing_outputs.cross_k_df,
        pending_scan_keys,
        _cross_k_summary_schema(cross_k_only=cross_k_only),
    )
    _write_csv_frame(_cross_k_summary_path(output_dir), cross_k_df)
    if cross_k_only:
        return

    modes_df = _drop_scan_rows(
        existing_outputs.modes_df,
        pending_scan_keys,
        MODES_SCHEMA,
    )
    _write_csv_frame(_modes_path(output_dir), modes_df)


def _closest_mode_to_zero(modes: np.ndarray) -> float | None:
    if modes.size == 0:
        return None
    return float(modes[np.argmin(np.abs(modes))])


def _count_shared_event_times(
    trade_event_times: np.ndarray,
    quote_event_times: np.ndarray,
) -> int:
    trade_index = 0
    quote_index = 0
    n_shared_event_times = 0

    while trade_index < trade_event_times.size and quote_index < quote_event_times.size:
        trade_time = trade_event_times[trade_index]
        quote_time = quote_event_times[quote_index]
        if trade_time < quote_time:
            trade_index += 1
            continue
        if trade_time > quote_time:
            quote_index += 1
            continue
        n_shared_event_times += 1
        trade_index += 1
        quote_index += 1

    return n_shared_event_times


def _scan_symbols(
    *,
    target_dates: Sequence[str],
    symbols: Sequence[str] | None,
    max_symbols: int | None,
) -> tuple[list[_SymbolScan], dict[str, list[str]]]:
    selected_symbols = set(symbols) if symbols is not None else None
    scans: list[_SymbolScan] = []
    symbols_by_date: dict[str, list[str]] = {}

    for date_yyyymmdd in target_dates:
        date_symbols = symbols_with_prefix(date_yyyymmdd, "")
        if selected_symbols is not None:
            date_symbols = [symbol for symbol in date_symbols if symbol in selected_symbols]
        if max_symbols is not None:
            date_symbols = date_symbols[:max_symbols]
        symbols_by_date[date_yyyymmdd] = date_symbols

        for symbol in date_symbols:
            trade_path = TRADE_BASE_DIR / symbol / f"trade_{date_yyyymmdd}.parquet"
            quote_path = QUOTE_BASE_DIR / symbol / f"quote_{date_yyyymmdd}.parquet"
            scans.append(
                _SymbolScan(
                    date_yyyymmdd=date_yyyymmdd,
                    symbol=symbol,
                    trade_path=trade_path,
                    quote_path=quote_path,
                    trade_counts=_scan_event_counts(
                        trade_path,
                        "Participant Timestamp",
                        obs_window=OBS_WINDOW,
                    ),
                    quote_counts=_scan_event_counts(
                        quote_path,
                        "Participant_Timestamp",
                        obs_window=OBS_WINDOW,
                    ),
                ),
            )

    if not scans:
        msg = "No symbols matched the requested batch configuration."
        raise ValueError(msg)

    return scans, symbols_by_date


def _run_id(
    *,
    target_dates: Sequence[str],
    min_events_per_side: int,
    created_at: datetime,
    subset_requested: bool,
    cross_k_only: bool,
) -> str:
    dates_token = "-".join(target_dates)
    timestamp_token = created_at.strftime("%Y%m%dT%H%M%SZ")
    subset_token = "_subset" if subset_requested else ""
    cross_k_only_token = "_crosskonly" if cross_k_only else ""
    return (
        f"{dates_token}_min{min_events_per_side}{subset_token}{cross_k_only_token}"
        f"_{timestamp_token}"
    )


def _resolve_output_dir(
    *,
    target_dates: Sequence[str],
    min_events_per_side: int,
    subset_requested: bool,
    cross_k_only: bool,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    created_at = datetime.now(UTC)
    run_dir = MODE_SUMMARY_OUTPUT_BASE_DIR / _run_id(
        target_dates=target_dates,
        min_events_per_side=min_events_per_side,
        created_at=created_at,
        subset_requested=subset_requested,
        cross_k_only=cross_k_only,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _scan_jobs(
    scans: Sequence[_SymbolScan],
    max_pairs: int | None,
) -> tuple[list[_ScanJob], int]:
    jobs: list[_ScanJob] = []
    total_progress_pairs = 0
    remaining_pairs = max_pairs

    for scan in scans:
        candidate_pairs = scan.candidate_pairs
        if remaining_pairs is not None:
            if remaining_pairs <= 0:
                break
            candidate_pairs = candidate_pairs[:remaining_pairs]
            remaining_pairs -= len(candidate_pairs)
        if not candidate_pairs:
            continue
        job = _ScanJob(scan=scan, candidate_pairs=tuple(candidate_pairs))
        jobs.append(job)
        total_progress_pairs += job.n_pairs

    return jobs, total_progress_pairs


def _is_completed_scan(
    scan_job: _ScanJob,
    existing_outputs: _ExistingOutputs,
    *,
    cross_k_only: bool,
) -> bool:
    # 再開時は expected pair がそろい、full mode では ok 行の mode 数も一致する
    # scan だけを完了扱いにする。
    for trade_exchange, quote_exchange in scan_job.candidate_pairs:
        pair_key = _pair_key(
            scan_job.scan.date_yyyymmdd,
            scan_job.scan.symbol,
            trade_exchange,
            quote_exchange,
        )
        pair_row = existing_outputs.cross_k_rows_by_key.get(pair_key)
        if pair_row is None:
            return False
        if cross_k_only or pair_row["status"] != "ok":
            continue
        expected_mode_count = int(pair_row["mode_count"] or 0)
        if existing_outputs.mode_counts_by_key.get(pair_key, 0) != expected_mode_count:
            return False
    return True


def _analyze_scan_job(
    *,
    scan_job: _ScanJob,
    min_events_per_side: int,
    cross_k_only: bool,
) -> _ScanResult:
    scan = scan_job.scan
    eligible_trade_exchanges = sorted(
        {
            trade_exchange
            for trade_exchange, _ in scan_job.candidate_pairs
            if scan.trade_counts[trade_exchange] >= min_events_per_side
        },
    )
    eligible_quote_exchanges = sorted(
        {
            quote_exchange
            for _, quote_exchange in scan_job.candidate_pairs
            if scan.quote_counts[quote_exchange] >= min_events_per_side
        },
    )
    trade_arrays = _event_arrays_by_exchange(
        scan.trade_path,
        "Participant Timestamp",
        eligible_trade_exchanges,
        obs_window=OBS_WINDOW,
    )
    quote_arrays = _event_arrays_by_exchange(
        scan.quote_path,
        "Participant_Timestamp",
        eligible_quote_exchanges,
        obs_window=OBS_WINDOW,
    )

    cross_k_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    for trade_exchange, quote_exchange in scan_job.candidate_pairs:
        trade_n_events = scan.trade_counts[trade_exchange]
        quote_n_events = scan.quote_counts[quote_exchange]
        cross_k_row: dict[str, object] = {
            "date_yyyymmdd": scan.date_yyyymmdd,
            "symbol": scan.symbol,
            "trade_exchange": trade_exchange,
            "quote_exchange": quote_exchange,
            "status": "",
            "n_trade_events": trade_n_events,
            "n_quote_events": quote_n_events,
            "n_shared_event_times": None,
            "elapsed_sec": None,
            "cross_k_neg_1e3_1e4": None,
            "cross_k_neg_1e4_1e5": None,
            "cross_k_neg_1e5_0": None,
            "cross_k_pos_0_1e5": None,
            "cross_k_pos_1e5_1e4": None,
            "cross_k_pos_1e4_1e3": None,
            "error_type": None,
            "error_message": None,
        }
        if not cross_k_only:
            cross_k_row["bandwidth"] = None
            cross_k_row["mode_count"] = None
            cross_k_row["closest_mode_to_zero_sec"] = None

        if trade_n_events < min_events_per_side or quote_n_events < min_events_per_side:
            cross_k_row["status"] = "skipped_min_events"
            cross_k_rows.append(cross_k_row)
            continue

        trade_event_times = trade_arrays[trade_exchange]
        quote_event_times = quote_arrays[quote_exchange]
        cross_k_row["n_shared_event_times"] = _count_shared_event_times(
            trade_event_times,
            quote_event_times,
        )

        start_time = time.perf_counter()
        try:
            if not cross_k_only:
                bandwidth = ppllag.lepski_bw_selector_for_cpcf_mode(
                    trade_event_times,
                    quote_event_times,
                    obs_window=OBS_WINDOW,
                    u_range=U_RANGE,
                    bw_candidates=BW_CANDIDATES.tolist(),
                    kernel=KERNEL,
                    allow_not_simple=ALLOW_NOT_SIMPLE,
                )
                modes = np.asarray(
                    ppllag.find_cpcf_modes(
                        trade_event_times,
                        quote_event_times,
                        u_range=U_RANGE,
                        obs_window=OBS_WINDOW,
                        bandwidth=bandwidth,
                        kernel=KERNEL,
                        allow_not_simple=ALLOW_NOT_SIMPLE,
                    ),
                    dtype=np.float64,
                )
            cross_k_values = {
                window_name: float(
                    ppllag.cross_k(
                        trade_event_times,
                        quote_event_times,
                        u_window=u_window,
                        obs_window=OBS_WINDOW,
                    ),
                )
                for window_name, u_window in CROSS_K_WINDOWS.items()
            }
            cross_k_row["status"] = "ok"
            if not cross_k_only:
                cross_k_row["bandwidth"] = float(bandwidth)
                cross_k_row["mode_count"] = int(modes.size)
                cross_k_row["closest_mode_to_zero_sec"] = _closest_mode_to_zero(modes)
            cross_k_row["elapsed_sec"] = time.perf_counter() - start_time
            cross_k_row.update(cross_k_values)
            if not cross_k_only:
                for mode_index, mode_sec in enumerate(modes.tolist()):
                    mode_rows.append(
                        {
                            "date_yyyymmdd": scan.date_yyyymmdd,
                            "symbol": scan.symbol,
                            "trade_exchange": trade_exchange,
                            "quote_exchange": quote_exchange,
                            "mode_index": mode_index,
                            "mode_sec": mode_sec,
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            cross_k_row["status"] = "error"
            cross_k_row["elapsed_sec"] = time.perf_counter() - start_time
            cross_k_row["error_type"] = type(exc).__name__
            cross_k_row["error_message"] = str(exc)

        cross_k_rows.append(cross_k_row)

    return _ScanResult(
        scan_key=scan_job.scan_key,
        cross_k_rows=cross_k_rows,
        mode_rows=mode_rows,
    )


def _scan_results(
    *,
    scan_jobs: Sequence[_ScanJob],
    min_events_per_side: int,
    n_jobs: int,
    cross_k_only: bool,
) -> Iterator[_ScanResult]:
    if n_jobs == 1:
        for scan_job in scan_jobs:
            yield _analyze_scan_job(
                scan_job=scan_job,
                min_events_per_side=min_events_per_side,
                cross_k_only=cross_k_only,
            )
        return

    parallel = Parallel(
        n_jobs=n_jobs,
        backend=JOBLIB_BACKEND,
        return_as="generator_unordered",
    )
    worker_jobs = (
        delayed(_analyze_scan_job)(
            scan_job=scan_job,
            min_events_per_side=min_events_per_side,
            cross_k_only=cross_k_only,
        )
        for scan_job in scan_jobs
    )
    yield from parallel(worker_jobs)


def _append_scan_result(
    output_dir: Path,
    scan_result: _ScanResult,
    *,
    cross_k_only: bool,
) -> None:
    # worker は計算だけにして、CSV 追記は main process に寄せて書き込み競合を避ける。
    _append_csv_rows(
        _cross_k_summary_path(output_dir),
        scan_result.cross_k_rows,
        _cross_k_summary_schema(cross_k_only=cross_k_only),
    )
    if not cross_k_only:
        _append_csv_rows(_modes_path(output_dir), scan_result.mode_rows, MODES_SCHEMA)


def build_mode_summary(
    *,
    target_dates: Sequence[str] = TARGET_DATES,
    symbols: Sequence[str] | None = None,
    max_symbols: int | None = None,
    max_pairs: int | None = None,
    min_events_per_side: int = MIN_EVENTS_PER_SIDE,
    n_jobs: int = N_JOBS,
    output_dir: Path | None = None,
    show_progress: bool = True,
    cross_k_only: bool = False,
) -> Path:
    """Run the trade-vs-quote batch study and update resumable CSV artifacts.

    Parameters
    ----------
    target_dates
        Trading dates formatted as ``YYYYMMDD``.
    symbols
        Optional exact symbol whitelist applied after loading all available
        symbols for each date.
    max_symbols
        Optional limit on the number of symbols per date, preserving sorted order.
    max_pairs
        Optional cap on the number of candidate pairs written to the output.
        Pairs are traversed in deterministic sorted order.
    min_events_per_side
        Minimum number of deduplicated events required on both trade and quote
        sides before a pair is analyzed.
    n_jobs
        Number of joblib worker processes. Use ``1`` for serial execution and
        ``-1`` to use all available cores.
    output_dir
        Output directory used for CSV append and resume. If omitted, create a new
        timestamped directory under ``data/derived/trade_quote_modes``.
    show_progress
        If True, show a ``tqdm`` progress bar over scheduled candidate pairs.
    cross_k_only
        If True, compute only the raw cross-K windows, omit mode estimation, and
        write only ``cross_k_summary.csv`` with the reduced cross-K-only schema.

    Returns
    -------
    pathlib.Path
        Output directory containing ``cross_k_summary.csv`` and, in full mode,
        ``modes.csv``.

    """
    scans, _ = _scan_symbols(
        target_dates=target_dates,
        symbols=symbols,
        max_symbols=max_symbols,
    )
    scan_jobs, _ = _scan_jobs(scans, max_pairs)
    run_dir = _resolve_output_dir(
        target_dates=target_dates,
        min_events_per_side=min_events_per_side,
        subset_requested=(symbols is not None or max_symbols is not None or max_pairs is not None),
        cross_k_only=cross_k_only,
        output_dir=output_dir,
    )
    _ensure_compatible_output_dir(run_dir, cross_k_only=cross_k_only)
    _ensure_output_files(run_dir, cross_k_only=cross_k_only)

    existing_outputs = _load_existing_outputs(run_dir, cross_k_only=cross_k_only)
    completed_scan_keys = {
        scan_job.scan_key
        for scan_job in scan_jobs
        if _is_completed_scan(
            scan_job,
            existing_outputs,
            cross_k_only=cross_k_only,
        )
    }
    pending_scan_jobs = [
        scan_job for scan_job in scan_jobs if scan_job.scan_key not in completed_scan_keys
    ]
    _cleanup_incomplete_scan_rows(
        run_dir,
        existing_outputs,
        {scan_job.scan_key for scan_job in pending_scan_jobs},
        cross_k_only=cross_k_only,
    )

    with tqdm(
        total=sum(scan_job.n_pairs for scan_job in pending_scan_jobs),
        desc="trade-quote pairs",
        disable=not show_progress,
    ) as progress_bar:
        for scan_result in _scan_results(
            scan_jobs=pending_scan_jobs,
            min_events_per_side=min_events_per_side,
            n_jobs=n_jobs,
            cross_k_only=cross_k_only,
        ):
            _append_scan_result(run_dir, scan_result, cross_k_only=cross_k_only)
            progress_bar.update(scan_result.n_pairs)

    return run_dir


def main() -> int:
    """Run the trade-vs-quote batch study from the command line.

    Returns
    -------
    int
        Process exit code.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        help="Trading date formatted as YYYYMMDD. Repeat to pass multiple dates.",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        help="Exact symbol to include. Repeat as needed.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Optional cap on the number of symbols per date.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on the number of candidate pairs written to the output.",
    )
    parser.add_argument(
        "--min-events-per-side",
        type=int,
        default=MIN_EVENTS_PER_SIDE,
        help="Minimum deduplicated event count required on both sides.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="Number of joblib worker processes. Use -1 for all available cores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory used for CSV append and resume.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--cross-k-only",
        action="store_true",
        help="Compute only raw cross-K windows and do not write modes.csv.",
    )
    args = parser.parse_args()

    run_dir = build_mode_summary(
        target_dates=tuple(args.dates) if args.dates else TARGET_DATES,
        symbols=tuple(args.symbols) if args.symbols else None,
        max_symbols=args.max_symbols,
        max_pairs=args.max_pairs,
        min_events_per_side=args.min_events_per_side,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        cross_k_only=args.cross_k_only,
    )
    summary = {
        "run_dir": str(run_dir),
        "cross_k_summary_csv": str(run_dir / CROSS_K_SUMMARY_FILENAME),
    }
    if args.cross_k_only:
        summary["cross_k_only"] = True
    else:
        summary["modes_csv"] = str(run_dir / MODES_FILENAME)
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
