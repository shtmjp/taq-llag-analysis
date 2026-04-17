from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from taq_llag_analysis import write_trade_quote_cpcf_examples as cpcf_examples
from taq_llag_analysis import write_trade_quote_mode_summary as mode_summary

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_build_cpcf_examples_reads_cross_k_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_id = "test-run"
    source_base_dir = tmp_path / "source"
    source_run_dir = source_base_dir / run_id
    source_run_dir.mkdir(parents=True)

    cross_k_rows = [
        {
            "date_yyyymmdd": "20251031",
            "symbol": "AAA",
            "trade_exchange": "Q",
            "quote_exchange": "Z",
            "status": "ok",
            "n_trade_events": 120,
            "n_quote_events": 130,
            "n_shared_event_times": 1,
            "bandwidth": 1.0e-6,
            "mode_count": 1,
            "closest_mode_to_zero_sec": 1.0e-6,
            "elapsed_sec": 0.01,
            "cross_k_neg_1e3_1e4": 0.1,
            "cross_k_neg_1e4_1e5": 0.1,
            "cross_k_neg_1e5_0": 0.1,
            "cross_k_pos_0_1e5": 0.1,
            "cross_k_pos_1e5_1e4": 0.1,
            "cross_k_pos_1e4_1e3": 0.1,
            "error_type": None,
            "error_message": None,
        },
    ]
    mode_rows = [
        {
            "date_yyyymmdd": "20251031",
            "symbol": "AAA",
            "trade_exchange": "Q",
            "quote_exchange": "Z",
            "mode_index": 0,
            "mode_sec": 1.0e-6,
        },
    ]
    pl.from_dicts(
        cross_k_rows,
        schema=mode_summary.CROSS_K_SUMMARY_SCHEMA,
    ).write_csv(source_run_dir / cpcf_examples.CROSS_K_SUMMARY_FILENAME)
    pl.from_dicts(
        mode_rows,
        schema=mode_summary.MODES_SCHEMA,
    ).write_csv(source_run_dir / cpcf_examples.MODES_FILENAME)

    def fake_event_arrays_by_exchange(
        path: Path,
        timestamp_col: str,
        exchanges: tuple[str, ...] | list[str],
        *,
        obs_window: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        del path, obs_window
        if timestamp_col == "Participant Timestamp":
            source = {"Q": np.array([1.0, 2.0], dtype=np.float64)}
        else:
            source = {"Z": np.array([2.0, 3.0], dtype=np.float64)}
        return {exchange: source[exchange] for exchange in exchanges}

    def fake_cpcf(
        data1: np.ndarray,
        data2: np.ndarray,
        u_values: np.ndarray,
        obs_window: tuple[float, float],
        *,
        bandwidth: float,
        kernel: str,
        allow_not_simple: bool,
    ) -> np.ndarray:
        del data1, data2, obs_window, bandwidth, kernel, allow_not_simple
        return np.zeros_like(u_values)

    def fake_plot_symbol(
        *,
        symbol: str,
        date_panels: tuple[dict[str, object], ...] | list[dict[str, object]],
        trade_exchange: str,
        quote_exchange: str,
        outpath: Path,
    ) -> Path:
        del symbol, date_panels, trade_exchange, quote_exchange
        outpath.write_text("placeholder", encoding="utf-8")
        return outpath

    monkeypatch.setattr(cpcf_examples, "MODE_SUMMARY_OUTPUT_BASE_DIR", source_base_dir)
    monkeypatch.setattr(cpcf_examples, "_event_arrays_by_exchange", fake_event_arrays_by_exchange)
    monkeypatch.setattr(cpcf_examples.ppllag, "cpcf", fake_cpcf)
    monkeypatch.setattr(cpcf_examples, "_plot_symbol", fake_plot_symbol)

    output_dir = cpcf_examples.build_cpcf_examples(
        run_id=run_id,
        symbols=("AAA",),
        dates=("20251031",),
        output_base_dir=tmp_path / "out",
    )

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert not hasattr(mode_summary, "U_GRID_SPEC")
    assert hasattr(cpcf_examples, "U_GRID_SPEC")
    assert run_manifest["cross_k_summary_csv"].endswith("cross_k_summary.csv")
    assert (output_dir / "cpcf__AAA.png").exists()
