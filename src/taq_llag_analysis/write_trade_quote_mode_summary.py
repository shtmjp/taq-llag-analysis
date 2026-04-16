from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import ppllag
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .preprocess.inventory import symbols_with_prefix

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence


TRADE_BASE_DIR = Path("data/filtered/trade")
QUOTE_BASE_DIR = Path("data/filtered/quote")
OUTPUT_BASE_DIR = Path("data/derived/trade_quote_modes")
TARGET_DATES: tuple[str, ...] = ("20251031", "20251103")
MIN_EVENTS_PER_SIDE = 100
N_JOBS = 1
OBS_WINDOW = (35_100.0, 56_700.0)
U_RANGE = (-1.0, 1.0)
BW_CANDIDATES = np.array([1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
U_GRID_SPEC: dict[str, object] = {
    "start": -1e-3,
    "stop": 1e-3,
    "num": 2001,
    "dtype": "float64",
}
KERNEL = "tent"
ALLOW_NOT_SIMPLE = False
JOBLIB_BACKEND = "loky"
CROSS_K_WINDOWS: dict[str, tuple[float, float]] = {
    "cross_k_neg_1e3_1e4": (-1e-3, -1e-4),
    "cross_k_neg_1e4_1e5": (-1e-4, -1e-5),
    "cross_k_neg_1e5_0": (-1e-5, 0.0),
    "cross_k_pos_0_1e5": (0.0, 1e-5),
    "cross_k_pos_1e5_1e4": (1e-5, 1e-4),
    "cross_k_pos_1e4_1e3": (1e-4, 1e-3),
}

SYMBOL_INVENTORY_SCHEMA: dict[str, pl.DataType] = {
    "date_yyyymmdd": pl.String,
    "symbol": pl.String,
    "trade_path_exists": pl.Boolean,
    "quote_path_exists": pl.Boolean,
    "n_trade_exchanges": pl.Int64,
    "n_quote_exchanges": pl.Int64,
    "n_candidate_pairs": pl.Int64,
    "n_pairs_ge_min_events": pl.Int64,
}
PAIR_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.String,
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
MODES_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.String,
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
    def trade_path_exists(self) -> bool:
        return self.trade_path.exists()

    @property
    def quote_path_exists(self) -> bool:
        return self.quote_path.exists()

    @property
    def candidate_pairs(self) -> list[tuple[str, str]]:
        return _candidate_pairs(self.trade_counts, self.quote_counts)

    def n_pairs_ge_min_events(self, min_events_per_side: int) -> int:
        return sum(
            1
            for trade_exchange, quote_exchange in self.candidate_pairs
            if self.trade_counts[trade_exchange] >= min_events_per_side
            and self.quote_counts[quote_exchange] >= min_events_per_side
        )


def _timestamp_to_time_expr(timestamp_col: str) -> pl.Expr:
    timestamp = pl.col(timestamp_col)
    hh = (timestamp // 10_000_000_000_000).cast(pl.Float64)
    mm = ((timestamp // 100_000_000_000) % 100).cast(pl.Float64)
    ss = ((timestamp // 1_000_000_000) % 100).cast(pl.Float64)
    subsec = (timestamp % 1_000_000_000).cast(pl.Float64) / 1_000_000_000.0
    return (hh * 3600.0 + mm * 60.0 + ss + subsec).alias("event_time")


def _event_time_frame(
    path: Path,
    timestamp_col: str,
    exchanges: Iterable[str] | None,
    *,
    obs_window: tuple[float, float],
) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema={"Exchange": pl.String, "event_time": pl.Float64})

    event_df = pl.read_parquet(path, columns=["Exchange", timestamp_col])
    if exchanges is not None:
        selected_exchanges = sorted(set(exchanges))
        if not selected_exchanges:
            return pl.DataFrame(schema={"Exchange": pl.String, "event_time": pl.Float64})
        event_df = event_df.filter(pl.col("Exchange").is_in(selected_exchanges))

    return (
        event_df.group_by(["Exchange", timestamp_col])
        .agg(pl.len().alias("n_rows"))
        .with_columns(_timestamp_to_time_expr(timestamp_col))
        .filter(pl.col("event_time").is_between(obs_window[0], obs_window[1]))
        .select(["Exchange", "event_time"])
        .sort(["Exchange", "event_time"])
    )


def _csv_frame(
    rows: list[dict[str, object]],
    schema: Mapping[str, pl.DataType],
) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.from_dicts(rows, schema=schema)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_output(args: Sequence[str]) -> str | None:
    completed = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata() -> dict[str, object]:
    git_sha = _git_output(("git", "rev-parse", "HEAD"))
    git_status = _git_output(("git", "status", "--short"))
    return {
        "commit_sha": git_sha,
        "dirty": bool(git_status),
    }


def _scan_event_counts(
    path: Path,
    timestamp_col: str,
    *,
    obs_window: tuple[float, float],
) -> dict[str, int]:
    counts_df = (
        _event_time_frame(
            path,
            timestamp_col,
            exchanges=None,
            obs_window=obs_window,
        )
        .group_by("Exchange")
        .agg(pl.len().alias("n_events"))
        .sort("Exchange")
    )
    return {exchange: int(n_events) for exchange, n_events in counts_df.iter_rows()}


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


def _event_arrays_by_exchange(
    path: Path,
    timestamp_col: str,
    exchanges: Iterable[str],
    *,
    obs_window: tuple[float, float],
) -> dict[str, np.ndarray]:
    selected_exchanges = sorted(set(exchanges))
    if not selected_exchanges:
        return {}

    event_df = _event_time_frame(
        path,
        timestamp_col,
        exchanges=selected_exchanges,
        obs_window=obs_window,
    )

    arrays_by_exchange: dict[str, np.ndarray] = {}
    for exchange_df in event_df.partition_by("Exchange", maintain_order=False):
        exchange = exchange_df.get_column("Exchange")[0]
        arrays_by_exchange[exchange] = np.asarray(
            exchange_df.get_column("event_time").to_numpy(),
            dtype=np.float64,
        )
    return arrays_by_exchange


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
) -> str:
    dates_token = "-".join(target_dates)
    timestamp_token = created_at.strftime("%Y%m%dT%H%M%SZ")
    subset_token = "_subset" if subset_requested else ""
    return f"{dates_token}_min{min_events_per_side}{subset_token}_{timestamp_token}"


def _scan_jobs(
    scans: Sequence[_SymbolScan],
    max_pairs: int | None,
) -> tuple[list[tuple[_SymbolScan, int]], int]:
    jobs: list[tuple[_SymbolScan, int]] = []
    total_progress_pairs = 0
    remaining_pairs = max_pairs

    for scan in scans:
        n_candidate_pairs = len(scan.candidate_pairs)
        if n_candidate_pairs == 0:
            continue
        if remaining_pairs is None:
            pair_limit = n_candidate_pairs
        else:
            if remaining_pairs <= 0:
                break
            pair_limit = min(n_candidate_pairs, remaining_pairs)
            remaining_pairs -= pair_limit
        jobs.append((scan, pair_limit))
        total_progress_pairs += pair_limit

    return jobs, total_progress_pairs


def _run_config_dict(
    *,
    run_id: str,
    created_at: datetime,
    target_dates: Sequence[str],
    min_events_per_side: int,
    n_jobs: int,
    max_symbols: int | None,
    max_pairs: int | None,
    selected_symbols: Sequence[str] | None,
    symbols_by_date: Mapping[str, Sequence[str]],
    scans: Sequence[_SymbolScan],
    scan_jobs: Sequence[tuple[_SymbolScan, int]],
) -> dict[str, object]:
    total_candidate_pairs = sum(len(scan.candidate_pairs) for scan in scans)
    total_eligible_pairs = sum(scan.n_pairs_ge_min_events(min_events_per_side) for scan in scans)
    total_scheduled_pairs = sum(pair_limit for _, pair_limit in scan_jobs)
    git_metadata = _git_metadata()
    return {
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "python_version": sys.version,
        "package_versions": {
            "joblib": _package_version("joblib"),
            "numpy": _package_version("numpy"),
            "polars": _package_version("polars"),
            "ppllag": _package_version("ppllag"),
            "tqdm": _package_version("tqdm"),
        },
        "git": git_metadata,
        "target_dates": list(target_dates),
        "selected_symbols": list(selected_symbols) if selected_symbols is not None else None,
        "symbols_by_date": {
            date_yyyymmdd: list(date_symbols)
            for date_yyyymmdd, date_symbols in symbols_by_date.items()
        },
        "max_symbols": max_symbols,
        "max_pairs": max_pairs,
        "trade_base_dir": str(TRADE_BASE_DIR),
        "quote_base_dir": str(QUOTE_BASE_DIR),
        "pair_rule": "trade_exchange != quote_exchange",
        "min_events_per_side": min_events_per_side,
        "n_jobs": n_jobs,
        "parallel_backend": JOBLIB_BACKEND if n_jobs != 1 else None,
        "obs_window": list(OBS_WINDOW),
        "u_range": list(U_RANGE),
        "bw_candidates": BW_CANDIDATES.tolist(),
        "kernel": KERNEL,
        "allow_not_simple": ALLOW_NOT_SIMPLE,
        "u_grid_spec": U_GRID_SPEC,
        "event_definitions": {
            "trade": (
                'filter one exchange, collapse on ("Exchange", "Participant Timestamp"), '
                "convert to float seconds, filter to obs_window, sort ascending"
            ),
            "quote": (
                'filter one exchange, collapse on ("Exchange", "Participant_Timestamp"), '
                "convert to float seconds, filter to obs_window, sort ascending"
            ),
        },
        "cross_k_windows": {
            window_name: list(window_values)
            for window_name, window_values in CROSS_K_WINDOWS.items()
        },
        "cross_k_note": "ppllag.cross_k is experimental in the installed package.",
        "inventory_summary": {
            "n_symbol_rows": len(scans),
            "n_candidate_pairs": total_candidate_pairs,
            "n_pairs_ge_min_events": total_eligible_pairs,
            "n_scheduled_pairs": total_scheduled_pairs,
        },
    }


def _analyze_scan(
    *,
    scan: _SymbolScan,
    run_id: str,
    min_events_per_side: int,
    pair_limit: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_pairs = scan.candidate_pairs
    if pair_limit is not None:
        candidate_pairs = candidate_pairs[:pair_limit]
    if not candidate_pairs:
        return [], []

    eligible_trade_exchanges = [
        exchange
        for exchange, n_events in scan.trade_counts.items()
        if n_events >= min_events_per_side
    ]
    eligible_quote_exchanges = [
        exchange
        for exchange, n_events in scan.quote_counts.items()
        if n_events >= min_events_per_side
    ]
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

    pair_rows: list[dict[str, object]] = []
    mode_rows: list[dict[str, object]] = []
    for trade_exchange, quote_exchange in candidate_pairs:
        trade_n_events = scan.trade_counts[trade_exchange]
        quote_n_events = scan.quote_counts[quote_exchange]
        pair_row: dict[str, object] = {
            "run_id": run_id,
            "date_yyyymmdd": scan.date_yyyymmdd,
            "symbol": scan.symbol,
            "trade_exchange": trade_exchange,
            "quote_exchange": quote_exchange,
            "status": "",
            "n_trade_events": trade_n_events,
            "n_quote_events": quote_n_events,
            "n_shared_event_times": None,
            "bandwidth": None,
            "mode_count": None,
            "closest_mode_to_zero_sec": None,
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

        if trade_n_events < min_events_per_side or quote_n_events < min_events_per_side:
            pair_row["status"] = "skipped_min_events"
            pair_rows.append(pair_row)
            continue

        trade_event_times = trade_arrays[trade_exchange]
        quote_event_times = quote_arrays[quote_exchange]
        n_shared_event_times = _count_shared_event_times(
            trade_event_times,
            quote_event_times,
        )
        pair_row["n_shared_event_times"] = n_shared_event_times
        if n_shared_event_times > 0 and not ALLOW_NOT_SIMPLE:
            pair_row["status"] = "skipped_not_simple"
            pair_row["error_type"] = "NotSimplePrecheck"
            pair_row["error_message"] = (
                f"trade and quote share {n_shared_event_times} event_time values"
            )
            pair_rows.append(pair_row)
            continue

        start_time = time.perf_counter()
        try:
            bandwidth = ppllag.lepski_bw_selector_for_cpcf_mode(
                trade_event_times,
                quote_event_times,
                obs_window=OBS_WINDOW,
                u_range=U_RANGE,
                bw_candidates=BW_CANDIDATES.tolist(),
                kernel=KERNEL,
                allow_not_simple=ALLOW_NOT_SIMPLE,
            )
            modes = ppllag.find_cpcf_modes(
                trade_event_times,
                quote_event_times,
                u_range=U_RANGE,
                obs_window=OBS_WINDOW,
                bandwidth=bandwidth,
                kernel=KERNEL,
                allow_not_simple=ALLOW_NOT_SIMPLE,
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
            pair_row["status"] = "ok"
            pair_row["bandwidth"] = float(bandwidth)
            pair_row["mode_count"] = int(modes.size)
            pair_row["closest_mode_to_zero_sec"] = _closest_mode_to_zero(modes)
            pair_row["elapsed_sec"] = time.perf_counter() - start_time
            pair_row.update(cross_k_values)
            for mode_index, mode_sec in enumerate(modes.tolist()):
                mode_rows.append(
                    {
                        "run_id": run_id,
                        "date_yyyymmdd": scan.date_yyyymmdd,
                        "symbol": scan.symbol,
                        "trade_exchange": trade_exchange,
                        "quote_exchange": quote_exchange,
                        "mode_index": mode_index,
                        "mode_sec": mode_sec,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            pair_row["status"] = "error"
            pair_row["elapsed_sec"] = time.perf_counter() - start_time
            pair_row["error_type"] = type(exc).__name__
            pair_row["error_message"] = str(exc)

        pair_rows.append(pair_row)

    return pair_rows, mode_rows


def _scan_results(
    *,
    scan_jobs: Sequence[tuple[_SymbolScan, int]],
    run_id: str,
    min_events_per_side: int,
    n_jobs: int,
) -> Iterator[tuple[list[dict[str, object]], list[dict[str, object]]]]:
    if n_jobs == 1:
        for scan, pair_limit in scan_jobs:
            yield _analyze_scan(
                scan=scan,
                run_id=run_id,
                min_events_per_side=min_events_per_side,
                pair_limit=pair_limit,
            )
        return

    parallel = Parallel(
        n_jobs=n_jobs,
        backend=JOBLIB_BACKEND,
        return_as="generator_unordered",
    )
    worker_jobs = (
        delayed(_analyze_scan)(
            scan=scan,
            run_id=run_id,
            min_events_per_side=min_events_per_side,
            pair_limit=pair_limit,
        )
        for scan, pair_limit in scan_jobs
    )
    yield from parallel(worker_jobs)


def build_mode_summary(
    *,
    target_dates: Sequence[str] = TARGET_DATES,
    symbols: Sequence[str] | None = None,
    max_symbols: int | None = None,
    max_pairs: int | None = None,
    min_events_per_side: int = MIN_EVENTS_PER_SIDE,
    n_jobs: int = N_JOBS,
    output_base_dir: Path = OUTPUT_BASE_DIR,
    show_progress: bool = True,
) -> Path:
    """Run the trade-vs-quote batch study and write CSV/JSON artifacts.

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
        Optional cap on the number of candidate pairs written to the batch output.
        Pairs are traversed in deterministic sorted order.
    min_events_per_side
        Minimum number of deduplicated events required on both trade and quote
        sides before a pair is analyzed.
    n_jobs
        Number of joblib worker processes. Use ``1`` for serial execution and
        ``-1`` to use all available cores.
    output_base_dir
        Parent directory for the batch run directory.
    show_progress
        If True, show a ``tqdm`` progress bar over candidate pairs.

    Returns
    -------
    pathlib.Path
        Output run directory containing the JSON/CSV artifacts.

    """
    scans, symbols_by_date = _scan_symbols(
        target_dates=target_dates,
        symbols=symbols,
        max_symbols=max_symbols,
    )
    scan_jobs, total_progress_pairs = _scan_jobs(scans, max_pairs)
    created_at = datetime.now(UTC)
    run_id = _run_id(
        target_dates=target_dates,
        min_events_per_side=min_events_per_side,
        created_at=created_at,
        subset_requested=(symbols is not None or max_symbols is not None or max_pairs is not None),
    )
    run_dir = output_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = _run_config_dict(
        run_id=run_id,
        created_at=created_at,
        target_dates=target_dates,
        min_events_per_side=min_events_per_side,
        n_jobs=n_jobs,
        max_symbols=max_symbols,
        max_pairs=max_pairs,
        selected_symbols=symbols,
        symbols_by_date=symbols_by_date,
        scans=scans,
        scan_jobs=scan_jobs,
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    symbol_inventory_rows: list[dict[str, object]] = []
    total_candidate_pairs = 0
    for scan in scans:
        n_candidate_pairs = len(scan.candidate_pairs)
        total_candidate_pairs += n_candidate_pairs
        symbol_inventory_rows.append(
            {
                "date_yyyymmdd": scan.date_yyyymmdd,
                "symbol": scan.symbol,
                "trade_path_exists": scan.trade_path_exists,
                "quote_path_exists": scan.quote_path_exists,
                "n_trade_exchanges": len(scan.trade_counts),
                "n_quote_exchanges": len(scan.quote_counts),
                "n_candidate_pairs": n_candidate_pairs,
                "n_pairs_ge_min_events": scan.n_pairs_ge_min_events(
                    min_events_per_side,
                ),
            },
        )

    symbol_inventory_df = _csv_frame(symbol_inventory_rows, SYMBOL_INVENTORY_SCHEMA).sort(
        ["date_yyyymmdd", "symbol"],
    )
    symbol_inventory_df.write_csv(run_dir / "symbol_inventory.csv")

    pair_summary_rows: list[dict[str, object]] = []
    modes_rows: list[dict[str, object]] = []

    with tqdm(
        total=total_progress_pairs,
        desc="trade-quote pairs",
        disable=not show_progress,
    ) as progress_bar:
        for scan_pair_rows, scan_mode_rows in _scan_results(
            scan_jobs=scan_jobs,
            run_id=run_id,
            min_events_per_side=min_events_per_side,
            n_jobs=n_jobs,
        ):
            pair_summary_rows.extend(scan_pair_rows)
            modes_rows.extend(scan_mode_rows)
            progress_bar.update(len(scan_pair_rows))

    pair_summary_df = _csv_frame(pair_summary_rows, PAIR_SUMMARY_SCHEMA).sort(
        ["date_yyyymmdd", "symbol", "trade_exchange", "quote_exchange"],
    )
    pair_summary_df.write_csv(run_dir / "pair_summary.csv")

    modes_df = _csv_frame(modes_rows, MODES_SCHEMA).sort(
        [
            "date_yyyymmdd",
            "symbol",
            "trade_exchange",
            "quote_exchange",
            "mode_index",
        ],
    )
    modes_df.write_csv(run_dir / "modes.csv")
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
        "--output-base-dir",
        type=Path,
        default=OUTPUT_BASE_DIR,
        help="Parent directory for the run output.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    args = parser.parse_args()

    run_dir = build_mode_summary(
        target_dates=tuple(args.dates) if args.dates else TARGET_DATES,
        symbols=tuple(args.symbols) if args.symbols else None,
        max_symbols=args.max_symbols,
        max_pairs=args.max_pairs,
        min_events_per_side=args.min_events_per_side,
        n_jobs=args.n_jobs,
        output_base_dir=args.output_base_dir,
        show_progress=not args.no_progress,
    )
    summary = {
        "run_dir": str(run_dir),
        "pair_summary_csv": str(run_dir / "pair_summary.csv"),
        "modes_csv": str(run_dir / "modes.csv"),
        "run_config_json": str(run_dir / "run_config.json"),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
