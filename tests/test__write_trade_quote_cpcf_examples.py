from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from taq_llag_analysis import write_trade_quote_cpcf_examples as cpcf_examples
from taq_llag_analysis import write_trade_quote_mode_summary as mode_summary

if TYPE_CHECKING:
    from pathlib import Path


def test_build_cpcf_examples_rejects_cross_k_only_source_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "cross_k_only_run"
    source_base_dir = tmp_path / "mode_runs"
    source_run_dir = source_base_dir / run_id
    source_run_dir.mkdir(parents=True)

    cross_k_df = pl.from_dicts(
        [
            {
                "date_yyyymmdd": "20251031",
                "symbol": "ABC",
                "trade_exchange": "Q",
                "quote_exchange": "Z",
                "status": "ok",
                "n_trade_events": 120,
                "n_quote_events": 110,
                "n_shared_event_times": 1,
                "elapsed_sec": 0.01,
                "cross_k_neg_1e3_1e4": -0.0011,
                "cross_k_neg_1e4_1e5": -0.00011,
                "cross_k_neg_1e5_0": -0.00001,
                "cross_k_pos_0_1e5": 0.00001,
                "cross_k_pos_1e5_1e4": 0.00011,
                "cross_k_pos_1e4_1e3": 0.0011,
                "error_type": None,
                "error_message": None,
            },
        ],
        schema=mode_summary.CROSS_K_ONLY_SUMMARY_SCHEMA,
    )
    cross_k_df.write_csv(source_run_dir / cpcf_examples.CROSS_K_SUMMARY_FILENAME)

    monkeypatch.setattr(cpcf_examples, "MODE_SUMMARY_OUTPUT_BASE_DIR", source_base_dir)

    with pytest.raises(ValueError, match="cross-K-only output"):
        cpcf_examples.build_cpcf_examples(
            run_id=run_id,
            symbols=("ABC",),
            dates=("20251031",),
            output_base_dir=tmp_path / "cpcf_out",
        )
