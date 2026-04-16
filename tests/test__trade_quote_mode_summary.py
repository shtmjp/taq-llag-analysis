from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from taq_llag_analysis import write_trade_quote_mode_summary as mode_summary

if TYPE_CHECKING:
    import pytest


def test_scan_symbols_loads_all_symbols_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_symbols_with_prefix(date_yyyymmdd: str, symbol_prefix: str) -> list[str]:
        calls.append((date_yyyymmdd, symbol_prefix))
        return ["AAA", "BBB"]

    def fake_scan_event_counts(
        path: Path,
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
        path: Path,
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


def test_run_config_omits_symbol_prefix() -> None:
    created_at = datetime(2026, 4, 13, tzinfo=UTC)
    scan = mode_summary._SymbolScan(  # noqa: SLF001
        date_yyyymmdd="20251031",
        symbol="AAA",
        trade_path=Path("trade.parquet"),
        quote_path=Path("quote.parquet"),
        trade_counts={"Q": 120},
        quote_counts={"Z": 130},
    )

    config = mode_summary._run_config_dict(  # noqa: SLF001
        run_id="20251031_min100_20260413T000000Z",
        created_at=created_at,
        target_dates=("20251031",),
        min_events_per_side=100,
        n_jobs=1,
        max_symbols=None,
        max_pairs=None,
        selected_symbols=("AAA",),
        symbols_by_date={"20251031": ["AAA"]},
        scans=(scan,),
        scan_jobs=((scan, 1),),
    )

    assert "symbol_prefix" not in config
    assert config["selected_symbols"] == ["AAA"]
    assert config["symbols_by_date"] == {"20251031": ["AAA"]}
