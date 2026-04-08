from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from write_m_trade_quote_mode_summary import (
    _count_shared_event_times,
    _csv_frame,
    _event_arrays_by_exchange,
    _scan_event_counts,
    build_mode_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


MIN_EVENTS_PER_SIDE = 100
N_JOBS = 2
N_SHARED_EVENT_TIMES = 2
CROSS_K_WINDOW_EXPECTED = {
    "cross_k_neg_1e3_1e4": [-0.001, -0.0001],
    "cross_k_neg_1e4_1e5": [-0.0001, -1e-05],
    "cross_k_neg_1e5_0": [-1e-05, 0.0],
    "cross_k_pos_0_1e5": [0.0, 1e-05],
    "cross_k_pos_1e5_1e4": [1e-05, 0.0001],
    "cross_k_pos_1e4_1e3": [0.0001, 0.001],
}


def test_build_mode_summary_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = build_mode_summary(
        target_dates=("20251103",),
        symbol_prefix="M",
        symbols=("MNDY",),
        n_jobs=N_JOBS,
        output_base_dir=tmp_path,
        show_progress=False,
    )

    run_config_path = run_dir / "run_config.json"
    symbol_inventory_path = run_dir / "symbol_inventory.csv"
    pair_summary_path = run_dir / "pair_summary.csv"
    modes_path = run_dir / "modes.csv"

    assert run_config_path.exists()
    assert symbol_inventory_path.exists()
    assert pair_summary_path.exists()
    assert modes_path.exists()

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert run_config["symbol_prefix"] == "M"
    assert run_config["target_dates"] == ["20251103"]
    assert run_config["min_events_per_side"] == MIN_EVENTS_PER_SIDE
    assert run_config["n_jobs"] == N_JOBS
    assert run_config["cross_k_windows"] == CROSS_K_WINDOW_EXPECTED

    symbol_inventory_df = pl.read_csv(symbol_inventory_path)
    pair_summary_df = pl.read_csv(pair_summary_path)
    modes_df = pl.read_csv(modes_path)

    assert symbol_inventory_df.height == 1
    assert {
        "date_yyyymmdd",
        "symbol",
        "trade_path_exists",
        "quote_path_exists",
        "n_trade_exchanges",
        "n_quote_exchanges",
        "n_candidate_pairs",
        "n_pairs_ge_min_events",
    } <= set(symbol_inventory_df.columns)

    assert pair_summary_df.height > 0
    assert {
        "run_id",
        "date_yyyymmdd",
        "symbol",
        "trade_exchange",
        "quote_exchange",
        "status",
        "n_trade_events",
        "n_quote_events",
        "n_shared_event_times",
        "bandwidth",
        "mode_count",
        "closest_mode_to_zero_sec",
        "elapsed_sec",
        "cross_k_neg_1e3_1e4",
        "cross_k_neg_1e4_1e5",
        "cross_k_neg_1e5_0",
        "cross_k_pos_0_1e5",
        "cross_k_pos_1e5_1e4",
        "cross_k_pos_1e4_1e3",
        "error_type",
        "error_message",
    } <= set(pair_summary_df.columns)

    ok_pairs_df = pair_summary_df.filter(pl.col("status") == "ok")
    assert ok_pairs_df.height > 0

    ok_row = ok_pairs_df.row(0, named=True)
    assert ok_row["bandwidth"] is not None
    assert ok_row["mode_count"] is not None
    assert ok_row["cross_k_neg_1e3_1e4"] is not None
    assert ok_row["cross_k_neg_1e4_1e5"] is not None
    assert ok_row["cross_k_neg_1e5_0"] is not None
    assert ok_row["cross_k_pos_0_1e5"] is not None
    assert ok_row["cross_k_pos_1e5_1e4"] is not None
    assert ok_row["cross_k_pos_1e4_1e3"] is not None

    assert modes_df.height > 0
    assert {
        "run_id",
        "date_yyyymmdd",
        "symbol",
        "trade_exchange",
        "quote_exchange",
        "mode_index",
        "mode_sec",
    } == set(modes_df.columns)


def test_scan_event_counts_and_arrays_clip_to_obs_window(tmp_path: Path) -> None:
    parquet_path = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "Exchange": ["A", "A", "A", "B", "B"],
            "Participant Timestamp": [
                94459000000000,
                94500000000000,
                200000000000000,
                94510000000000,
                94510000000000,
            ],
        },
    ).write_parquet(parquet_path)

    counts = _scan_event_counts(
        parquet_path,
        "Participant Timestamp",
        obs_window=(35_100.0, 56_700.0),
    )
    arrays = _event_arrays_by_exchange(
        parquet_path,
        "Participant Timestamp",
        exchanges=["A", "B"],
        obs_window=(35_100.0, 56_700.0),
    )

    assert counts == {"A": 1, "B": 1}
    assert np.array_equal(arrays["A"], np.array([35_100.0]))
    assert np.array_equal(arrays["B"], np.array([35_110.0]))


def test_count_shared_event_times_counts_exact_overlaps() -> None:
    shared_count = _count_shared_event_times(
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([2.0, 3.0, 4.0, 9.0]),
    )

    assert shared_count == N_SHARED_EVENT_TIMES


def test_csv_frame_uses_declared_schema_for_late_string_values() -> None:
    df = _csv_frame(
        [
            {"value": 1.0, "error_type": None, "error_message": None},
            {
                "value": 2.0,
                "error_type": "NotSimplePrecheck",
                "error_message": "trade and quote share 3 event_time values",
            },
        ],
        schema={
            "value": pl.Float64,
            "error_type": pl.String,
            "error_message": pl.String,
        },
    )

    assert df.schema == {
        "value": pl.Float64,
        "error_type": pl.String,
        "error_message": pl.String,
    }
    assert df.row(1, named=True)["error_type"] == "NotSimplePrecheck"
