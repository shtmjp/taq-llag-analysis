from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from taq_llag_analysis import write_trade_quote_mode_summary as mode_summary

MODE_VALUES = [-2e-6, 3e-6]


def _single_scan() -> object:
    scan_cls = mode_summary._SymbolScan  # noqa: SLF001
    return scan_cls(
        date_yyyymmdd="20251031",
        symbol="ABC",
        trade_path=Path("trade.parquet"),
        quote_path=Path("quote.parquet"),
        trade_counts={"Q": 120},
        quote_counts={"Z": 110},
    )


def _install_single_scan_stubs(monkeypatch: pytest.MonkeyPatch) -> object:
    scan = _single_scan()
    monkeypatch.setattr(
        mode_summary,
        "_scan_symbols",
        lambda **_: ([scan], {scan.date_yyyymmdd: [scan.symbol]}),
    )

    def fake_event_arrays_by_exchange(
        path: Path,
        timestamp_col: str,
        exchanges: list[str],
        *,
        obs_window: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        del path, exchanges, obs_window
        if timestamp_col == "Participant Timestamp":
            return {"Q": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        return {"Z": np.array([1.0, 4.0, 5.0], dtype=np.float64)}

    monkeypatch.setattr(mode_summary, "_event_arrays_by_exchange", fake_event_arrays_by_exchange)
    return scan


def _install_cross_k_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_cross_k(
        trade_event_times: np.ndarray,
        quote_event_times: np.ndarray,
        *,
        u_window: tuple[float, float],
        obs_window: tuple[float, float],
    ) -> float:
        del trade_event_times, quote_event_times, obs_window
        return float(sum(u_window))

    monkeypatch.setattr(mode_summary.ppllag, "cross_k", fake_cross_k)


def test_build_mode_summary_default_writes_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan = _install_single_scan_stubs(monkeypatch)
    _install_cross_k_stub(monkeypatch)

    def fake_bandwidth(*_args: object, **_kwargs: object) -> float:
        return 1e-6

    def fake_modes(*_args: object, **_kwargs: object) -> list[float]:
        return MODE_VALUES

    monkeypatch.setattr(mode_summary.ppllag, "lepski_bw_selector_for_cpcf_mode", fake_bandwidth)
    monkeypatch.setattr(mode_summary.ppllag, "find_cpcf_modes", fake_modes)

    run_dir = mode_summary.build_mode_summary(
        target_dates=(scan.date_yyyymmdd,),
        output_dir=tmp_path / "full_run",
        show_progress=False,
    )

    cross_k_df = pl.read_csv(run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME)
    modes_df = pl.read_csv(run_dir / mode_summary.MODES_FILENAME)

    assert cross_k_df.columns == list(mode_summary.CROSS_K_SUMMARY_SCHEMA)
    assert cross_k_df.height == 1
    row = cross_k_df.row(0, named=True)
    assert row["status"] == "ok"
    assert row["bandwidth"] == pytest.approx(1e-6)
    assert row["mode_count"] == len(MODE_VALUES)
    assert row["closest_mode_to_zero_sec"] == pytest.approx(-2e-6)
    assert row["n_shared_event_times"] == 1
    assert row["cross_k_neg_1e3_1e4"] == pytest.approx(-0.0011)
    assert row["cross_k_pos_1e4_1e3"] == pytest.approx(0.0011)
    assert row["elapsed_sec"] is not None

    assert modes_df.height == len(MODE_VALUES)
    assert modes_df.get_column("mode_sec").to_list() == MODE_VALUES


def test_build_mode_summary_cross_k_only_skips_mode_estimation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan = _install_single_scan_stubs(monkeypatch)
    _install_cross_k_stub(monkeypatch)

    def fail_mode_estimation(*_args: object, **_kwargs: object) -> float:
        message = "mode estimation should not run in cross-K-only mode"
        raise AssertionError(message)

    monkeypatch.setattr(
        mode_summary.ppllag,
        "lepski_bw_selector_for_cpcf_mode",
        fail_mode_estimation,
    )
    monkeypatch.setattr(mode_summary.ppllag, "find_cpcf_modes", fail_mode_estimation)

    run_dir = mode_summary.build_mode_summary(
        target_dates=(scan.date_yyyymmdd,),
        output_dir=tmp_path / "cross_k_only_run",
        show_progress=False,
        cross_k_only=True,
    )

    cross_k_df = pl.read_csv(run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME)

    assert cross_k_df.columns == list(mode_summary.CROSS_K_ONLY_SUMMARY_SCHEMA)
    assert not (run_dir / mode_summary.MODES_FILENAME).exists()
    row = cross_k_df.row(0, named=True)
    assert row["status"] == "ok"
    assert row["n_shared_event_times"] == 1
    assert row["cross_k_neg_1e3_1e4"] == pytest.approx(-0.0011)
    assert row["cross_k_pos_1e4_1e3"] == pytest.approx(0.0011)
    assert row["elapsed_sec"] is not None


def test_build_mode_summary_cross_k_only_resume_uses_existing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan = _install_single_scan_stubs(monkeypatch)
    _install_cross_k_stub(monkeypatch)

    run_dir = mode_summary.build_mode_summary(
        target_dates=(scan.date_yyyymmdd,),
        output_dir=tmp_path / "resume_cross_k_only",
        show_progress=False,
        cross_k_only=True,
    )

    def fail_if_recomputed(*_args: object, **_kwargs: object) -> float:
        message = "resume should not recompute completed cross-K-only rows"
        raise AssertionError(message)

    monkeypatch.setattr(mode_summary, "_event_arrays_by_exchange", fail_if_recomputed)
    monkeypatch.setattr(mode_summary.ppllag, "cross_k", fail_if_recomputed)

    rerun_dir = mode_summary.build_mode_summary(
        target_dates=(scan.date_yyyymmdd,),
        output_dir=run_dir,
        show_progress=False,
        cross_k_only=True,
    )

    assert rerun_dir == run_dir
    cross_k_df = pl.read_csv(run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME)
    assert cross_k_df.height == 1


@pytest.mark.parametrize(
    ("first_mode_name", "second_mode_name"),
    [("full", "cross_k_only"), ("cross_k_only", "full")],
)
def test_build_mode_summary_rejects_mixed_output_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_mode_name: str,
    second_mode_name: str,
) -> None:
    scan = _install_single_scan_stubs(monkeypatch)
    _install_cross_k_stub(monkeypatch)

    def fake_bandwidth(*_args: object, **_kwargs: object) -> float:
        return 1e-6

    def fake_modes(*_args: object, **_kwargs: object) -> list[float]:
        return MODE_VALUES

    monkeypatch.setattr(mode_summary.ppllag, "lepski_bw_selector_for_cpcf_mode", fake_bandwidth)
    monkeypatch.setattr(mode_summary.ppllag, "find_cpcf_modes", fake_modes)

    first_cross_k_only = first_mode_name == "cross_k_only"
    second_cross_k_only = second_mode_name == "cross_k_only"
    output_dir = tmp_path / f"mixed_{first_mode_name}_{second_mode_name}"
    mode_summary.build_mode_summary(
        target_dates=(scan.date_yyyymmdd,),
        output_dir=output_dir,
        show_progress=False,
        cross_k_only=first_cross_k_only,
    )

    with pytest.raises(ValueError, match=r"incompatible|cannot be reused"):
        mode_summary.build_mode_summary(
            target_dates=(scan.date_yyyymmdd,),
            output_dir=output_dir,
            show_progress=False,
            cross_k_only=second_cross_k_only,
        )


def test_main_cross_k_only_summary_omits_modes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "cli_run"

    def fake_build_mode_summary(**_: object) -> Path:
        return run_dir

    monkeypatch.setattr(mode_summary, "build_mode_summary", fake_build_mode_summary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "write_trade_quote_mode_summary.py",
            "--cross-k-only",
            "--output-dir",
            str(run_dir),
        ],
    )

    exit_code = mode_summary.main()

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["run_dir"] == str(run_dir)
    assert summary["cross_k_summary_csv"] == str(run_dir / mode_summary.CROSS_K_SUMMARY_FILENAME)
    assert summary["cross_k_only"] is True
    assert "modes_csv" not in summary


def test_run_id_marks_cross_k_only() -> None:
    run_id = mode_summary._run_id(  # noqa: SLF001
        target_dates=("20251031",),
        min_events_per_side=mode_summary.MIN_EVENTS_PER_SIDE,
        created_at=mode_summary.datetime(2026, 4, 20, tzinfo=mode_summary.UTC),
        subset_requested=False,
        cross_k_only=True,
    )

    assert "_crosskonly_" in run_id
