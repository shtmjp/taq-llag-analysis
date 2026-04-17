from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from taq_llag_analysis import write_trade_quote_mode_summary as mode_summary

if TYPE_CHECKING:
    import pytest


EXPECTED_MODE_COUNT = 2


def test_scan_symbols_loads_all_symbols_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_symbols_with_prefix(date_yyyymmdd: str, symbol_prefix: str) -> list[str]:
        calls.append((date_yyyymmdd, symbol_prefix))
        return ["AAA", "BBB"]

    def fake_scan_event_counts(
        path: pathlib.Path,
        timestamp_col: str,
        *,
        obs_window: tuple[float, float],
    ) -> dict[str, int]:
        del path, timestamp_col, obs_window
        return {"Q": 1}

    monkeypatch.setattr(mode_summary, "symbols_with_prefix", fake_symbols_with_prefix)
    monkeypatch.setattr(mode_summary, "_scan_event_counts", fake_scan_event_counts)

    scans, symbols_by_date = mode_summary._scan_symbols(  # noqa: SLF001
        target_dates=("20251031",),
        symbols=None,
        max_symbols=None,
    )

    assert calls == [("20251031", "")]
    assert symbols_by_date == {"20251031": ["AAA", "BBB"]}
    assert [scan.symbol for scan in scans] == ["AAA", "BBB"]


def test_scan_symbols_applies_exact_symbol_whitelist_after_loading_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_symbols_with_prefix(date_yyyymmdd: str, symbol_prefix: str) -> list[str]:
        del date_yyyymmdd, symbol_prefix
        return ["AAA", "BBB", "CCC"]

    def fake_scan_event_counts(
        path: pathlib.Path,
        timestamp_col: str,
        *,
        obs_window: tuple[float, float],
    ) -> dict[str, int]:
        del path, timestamp_col, obs_window
        return {"Q": 1}

    monkeypatch.setattr(
        mode_summary,
        "symbols_with_prefix",
        fake_symbols_with_prefix,
    )
    monkeypatch.setattr(mode_summary, "_scan_event_counts", fake_scan_event_counts)

    scans, symbols_by_date = mode_summary._scan_symbols(  # noqa: SLF001
        target_dates=("20251031",),
        symbols=("CCC", "AAA"),
        max_symbols=None,
    )

    assert symbols_by_date == {"20251031": ["AAA", "CCC"]}
    assert [scan.symbol for scan in scans] == ["AAA", "CCC"]


def test_run_id_uses_date_prefix_without_leading_token() -> None:
    created_at = datetime(2026, 4, 13, 7, 8, 9, tzinfo=UTC)

    run_id = mode_summary._run_id(  # noqa: SLF001
        target_dates=("20251031", "20251103"),
        min_events_per_side=100,
        created_at=created_at,
        subset_requested=False,
    )
    subset_run_id = mode_summary._run_id(  # noqa: SLF001
        target_dates=("20251031", "20251103"),
        min_events_per_side=100,
        created_at=created_at,
        subset_requested=True,
    )

    assert run_id == "20251031-20251103_min100_20260413T070809Z"
    assert subset_run_id == "20251031-20251103_min100_subset_20260413T070809Z"


def _patch_single_symbol_scan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trade_counts: dict[str, int] | None = None,
    quote_counts: dict[str, int] | None = None,
) -> None:
    trade_counts = trade_counts or {"Q": 120}
    quote_counts = quote_counts or {"Z": 130}

    def fake_symbols_with_prefix(date_yyyymmdd: str, symbol_prefix: str) -> list[str]:
        del date_yyyymmdd, symbol_prefix
        return ["AAA"]

    def fake_scan_event_counts(
        path: pathlib.Path,
        timestamp_col: str,
        *,
        obs_window: tuple[float, float],
    ) -> dict[str, int]:
        del path, obs_window
        if timestamp_col == "Participant Timestamp":
            return trade_counts
        return quote_counts

    monkeypatch.setattr(mode_summary, "symbols_with_prefix", fake_symbols_with_prefix)
    monkeypatch.setattr(mode_summary, "_scan_event_counts", fake_scan_event_counts)


def _patch_analysis_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trade_arrays: dict[str, np.ndarray] | None = None,
    quote_arrays: dict[str, np.ndarray] | None = None,
    modes: np.ndarray | None = None,
) -> None:
    trade_arrays = trade_arrays or {"Q": np.array([1.0, 2.0], dtype=np.float64)}
    quote_arrays = quote_arrays or {"Z": np.array([2.0, 3.0], dtype=np.float64)}
    modes = modes if modes is not None else np.array([-2.0e-6, 1.0e-6], dtype=np.float64)

    def fake_event_arrays_by_exchange(
        path: pathlib.Path,
        timestamp_col: str,
        exchanges: tuple[str, ...] | list[str],
        *,
        obs_window: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        del path, obs_window
        source = trade_arrays if timestamp_col == "Participant Timestamp" else quote_arrays
        return {exchange: np.asarray(source[exchange], dtype=np.float64) for exchange in exchanges}

    def fake_lepski_bw_selector_for_cpcf_mode(*args: object, **kwargs: object) -> float:
        del args, kwargs
        return 1.0e-6

    def fake_find_cpcf_modes(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return np.asarray(modes, dtype=np.float64)

    def fake_cross_k(
        data1: np.ndarray,
        data2: np.ndarray,
        *,
        u_window: tuple[float, float],
        obs_window: tuple[float, float],
    ) -> float:
        del data1, data2, obs_window
        return float(u_window[1] - u_window[0])

    monkeypatch.setattr(mode_summary, "_event_arrays_by_exchange", fake_event_arrays_by_exchange)
    monkeypatch.setattr(
        mode_summary.ppllag,
        "lepski_bw_selector_for_cpcf_mode",
        fake_lepski_bw_selector_for_cpcf_mode,
    )
    monkeypatch.setattr(mode_summary.ppllag, "find_cpcf_modes", fake_find_cpcf_modes)
    monkeypatch.setattr(mode_summary.ppllag, "cross_k", fake_cross_k)


def _cross_k_ok_row(*, mode_count: int) -> dict[str, object]:
    return {
        "date_yyyymmdd": "20251031",
        "symbol": "AAA",
        "trade_exchange": "Q",
        "quote_exchange": "Z",
        "status": "ok",
        "n_trade_events": 120,
        "n_quote_events": 130,
        "n_shared_event_times": 1,
        "bandwidth": 1.0e-6,
        "mode_count": mode_count,
        "closest_mode_to_zero_sec": 1.0e-6 if mode_count else None,
        "elapsed_sec": 0.01,
        "cross_k_neg_1e3_1e4": 0.1,
        "cross_k_neg_1e4_1e5": 0.1,
        "cross_k_neg_1e5_0": 0.1,
        "cross_k_pos_0_1e5": 0.1,
        "cross_k_pos_1e5_1e4": 0.1,
        "cross_k_pos_1e4_1e3": 0.1,
        "error_type": None,
        "error_message": None,
    }


def test_analyze_scan_job_allows_shared_event_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_analysis_backend(monkeypatch)
    scan = mode_summary._SymbolScan(  # noqa: SLF001
        date_yyyymmdd="20251031",
        symbol="AAA",
        trade_path=pathlib.Path("trade.parquet"),
        quote_path=pathlib.Path("quote.parquet"),
        trade_counts={"Q": 120},
        quote_counts={"Z": 130},
    )
    scan_job = mode_summary._ScanJob(  # noqa: SLF001
        scan=scan,
        candidate_pairs=(("Q", "Z"),),
    )

    result = mode_summary._analyze_scan_job(  # noqa: SLF001
        scan_job=scan_job,
        min_events_per_side=100,
    )

    assert result.cross_k_rows[0]["status"] == "ok"
    assert result.cross_k_rows[0]["n_shared_event_times"] == 1


def test_build_mode_summary_writes_cross_k_and_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    _patch_single_symbol_scan(monkeypatch)
    _patch_analysis_backend(monkeypatch)

    run_dir = mode_summary.build_mode_summary(
        target_dates=("20251031",),
        output_dir=tmp_path / "run",
        show_progress=False,
    )

    cross_k_path = run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME
    modes_path = run_dir / mode_summary.MODES_FILENAME
    assert cross_k_path.exists()
    assert modes_path.exists()
    assert not (run_dir / "pair_summary.csv").exists()
    assert not (run_dir / "run_config.json").exists()
    assert not (run_dir / "symbol_inventory.csv").exists()

    cross_k_df = pl.read_csv(cross_k_path, schema_overrides=mode_summary.CROSS_K_SUMMARY_SCHEMA)
    modes_df = pl.read_csv(modes_path, schema_overrides=mode_summary.MODES_SCHEMA)

    assert "run_id" not in cross_k_df.columns
    assert "run_id" not in modes_df.columns
    assert cross_k_df.height == 1
    assert int(cross_k_df.row(0, named=True)["mode_count"]) == EXPECTED_MODE_COUNT
    assert modes_df.height == EXPECTED_MODE_COUNT


def test_build_mode_summary_skips_completed_scan_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    _patch_single_symbol_scan(monkeypatch)
    _patch_analysis_backend(monkeypatch)
    run_dir = tmp_path / "run"

    mode_summary.build_mode_summary(
        target_dates=("20251031",),
        output_dir=run_dir,
        show_progress=False,
    )

    def should_not_run(*_args: object, **_kwargs: object) -> object:
        message = "completed scan should be skipped on resume"
        raise AssertionError(message)

    monkeypatch.setattr(mode_summary, "_analyze_scan_job", should_not_run)

    mode_summary.build_mode_summary(
        target_dates=("20251031",),
        output_dir=run_dir,
        show_progress=False,
    )

    cross_k_df = pl.read_csv(
        run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME,
        schema_overrides=mode_summary.CROSS_K_SUMMARY_SCHEMA,
    )
    modes_df = pl.read_csv(
        run_dir / mode_summary.MODES_FILENAME,
        schema_overrides=mode_summary.MODES_SCHEMA,
    )
    assert cross_k_df.height == 1
    assert modes_df.height == EXPECTED_MODE_COUNT


def test_build_mode_summary_treats_zero_mode_ok_row_as_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    _patch_single_symbol_scan(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.from_dicts(
        [_cross_k_ok_row(mode_count=0)],
        schema=mode_summary.CROSS_K_SUMMARY_SCHEMA,
    ).write_csv(run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME)
    pl.DataFrame(schema=mode_summary.MODES_SCHEMA).write_csv(run_dir / mode_summary.MODES_FILENAME)

    def should_not_run(*_args: object, **_kwargs: object) -> object:
        message = "mode_count=0 ok row should be considered completed"
        raise AssertionError(message)

    monkeypatch.setattr(mode_summary, "_analyze_scan_job", should_not_run)

    mode_summary.build_mode_summary(
        target_dates=("20251031",),
        output_dir=run_dir,
        show_progress=False,
    )


def test_build_mode_summary_cleans_incomplete_scan_before_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    _patch_single_symbol_scan(monkeypatch)
    _patch_analysis_backend(
        monkeypatch,
        modes=np.array([1.0e-6], dtype=np.float64),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.from_dicts(
        [_cross_k_ok_row(mode_count=1)],
        schema=mode_summary.CROSS_K_SUMMARY_SCHEMA,
    ).write_csv(run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME)
    pl.DataFrame(schema=mode_summary.MODES_SCHEMA).write_csv(run_dir / mode_summary.MODES_FILENAME)

    mode_summary.build_mode_summary(
        target_dates=("20251031",),
        output_dir=run_dir,
        show_progress=False,
    )

    cross_k_df = pl.read_csv(
        run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME,
        schema_overrides=mode_summary.CROSS_K_SUMMARY_SCHEMA,
    )
    modes_df = pl.read_csv(
        run_dir / mode_summary.MODES_FILENAME,
        schema_overrides=mode_summary.MODES_SCHEMA,
    )

    assert cross_k_df.height == 1
    assert modes_df.height == 1
