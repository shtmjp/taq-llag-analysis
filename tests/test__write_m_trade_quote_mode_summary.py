from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl

from write_m_trade_quote_mode_summary import build_mode_summary

if TYPE_CHECKING:
    from pathlib import Path


MIN_EVENTS_PER_SIDE = 100
N_JOBS = 2


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
    assert run_config["cross_k_windows"]["cross_k_pos_0_1e3"] == [0.0, 0.001]

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
        "bandwidth",
        "mode_count",
        "closest_mode_to_zero_sec",
        "elapsed_sec",
        "cross_k_neg_1e3_0",
        "cross_k_neg_1e4_0",
        "cross_k_pos_0_1e4",
        "cross_k_pos_0_1e3",
        "error_type",
        "error_message",
    } <= set(pair_summary_df.columns)

    ok_pairs_df = pair_summary_df.filter(pl.col("status") == "ok")
    assert ok_pairs_df.height > 0

    ok_row = ok_pairs_df.row(0, named=True)
    assert ok_row["bandwidth"] is not None
    assert ok_row["mode_count"] is not None
    assert ok_row["cross_k_neg_1e3_0"] is not None
    assert ok_row["cross_k_neg_1e4_0"] is not None
    assert ok_row["cross_k_pos_0_1e4"] is not None
    assert ok_row["cross_k_pos_0_1e3"] is not None

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
