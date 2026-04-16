from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from taq_llag_analysis.preprocess import (
    daily_taq_paths,
    inventory,
    quote_cli,
    quote_logic,
    quote_pipeline,
    schemas,
    trade_cli,
    trade_logic,
    trade_pipeline,
)
from taq_llag_analysis.preprocess.write_filtered_quote_parquet import write_filtered_quote_parquets
from taq_llag_analysis.preprocess.write_filtered_trade_parquet import write_filtered_trade_parquets

if TYPE_CHECKING:
    import pytest


def _write_pipe_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["|".join(columns)]
    lines.extend(
        "|".join("" if (value := row.get(column, "")) is None else str(value) for column in columns)
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trade_row(**values: object) -> dict[str, object]:
    return {column: values.get(column, "") for column in schemas.TRADE_RAW_COLUMNS}


def _quote_row(**values: object) -> dict[str, object]:
    return {column: values.get(column, "") for column in schemas.QUOTE_RAW_COLUMNS}


def _redirect_output_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(daily_taq_paths, "TRADE_OUTPUT_DIR", tmp_path / "trade")
    monkeypatch.setattr(daily_taq_paths, "QUOTE_OUTPUT_DIR", tmp_path / "quote")
    monkeypatch.setattr(
        daily_taq_paths,
        "TRADE_AUDIT_DIR",
        daily_taq_paths.TRADE_OUTPUT_DIR / "_audit",
    )
    monkeypatch.setattr(
        daily_taq_paths,
        "QUOTE_AUDIT_DIR",
        daily_taq_paths.QUOTE_OUTPUT_DIR / "_audit",
    )


def test_trade_logic_conditions_are_ordered() -> None:
    conditions = trade_logic.filter_conditions(symbols=["AAA"], exchanges=["N"])

    assert [name for name, _ in conditions] == ["symbol", "exchange", "market_hours"]
    assert trade_logic.filter_required_columns() == (
        "Symbol",
        "Exchange",
        "Participant Timestamp",
    )


def test_trade_logic_filters_expected_rows() -> None:
    trade_df = pl.DataFrame(
        {
            "Symbol": ["AAA", "AAA", "BBB"],
            "Exchange": ["N", "Q", "N"],
            "Participant Timestamp": [94_500_000_000_000, 94_500_000_000_000, 90_000_000_000_000],
        },
    )

    filtered_df = trade_logic.apply_trade_filters(
        trade_df.lazy(),
        symbols=["AAA"],
        exchanges=["N"],
    ).collect()

    assert filtered_df.to_dict(as_series=False) == {
        "Symbol": ["AAA"],
        "Exchange": ["N"],
        "Participant Timestamp": [94_500_000_000_000],
    }


def test_quote_logic_conditions_are_ordered() -> None:
    conditions = quote_logic.filter_conditions(symbols=["AAA"], exchanges=["N"])

    assert [name for name, _ in conditions] == [
        "symbol",
        "exchange",
        "positive_bid_offer",
        "positive_spread",
        "market_hours",
    ]
    assert quote_logic.filter_required_columns() == (
        "Symbol",
        "Exchange",
        "Bid_Price",
        "Offer_Price",
        "Time",
    )


def test_quote_logic_filters_expected_rows() -> None:
    quote_df = pl.DataFrame(
        {
            "Symbol": ["AAA", "AAA", "AAA", "BBB"],
            "Exchange": ["N", "N", "N", "N"],
            "Bid_Price": [10.0, 0.0, 10.2, 10.0],
            "Offer_Price": [10.1, 10.1, 10.1, 10.1],
            "Time": [
                94_500_000_000_000,
                94_500_000_000_000,
                94_500_000_000_000,
                57_000_000_000_000,
            ],
        },
    )

    filtered_df = quote_logic.apply_quote_filters(
        quote_df.lazy(),
        symbols=["AAA"],
        exchanges=["N"],
    ).collect()

    assert filtered_df.to_dict(as_series=False) == {
        "Symbol": ["AAA"],
        "Exchange": ["N"],
        "Bid_Price": [10.0],
        "Offer_Price": [10.1],
        "Time": [94_500_000_000_000],
    }


def test_trade_pipeline_appends_symbol_for_partitioning(monkeypatch: pytest.MonkeyPatch) -> None:
    trade_df = pl.DataFrame(
        {
            "Symbol": ["AAA"],
            "Exchange": ["N"],
            "Participant Timestamp": [94_500_000_000_000],
        },
    )
    monkeypatch.setattr(trade_pipeline, "scan_trade_lazy_frame", lambda _: trade_df.lazy())

    selected_df = trade_pipeline.selected_trade_lazy_frame(
        Path("ignored"),
        symbols=["AAA"],
        exchanges=None,
        columns=("Exchange", "Participant Timestamp"),
    ).collect()

    assert selected_df.columns == ["Exchange", "Participant Timestamp", "Symbol"]


def test_quote_pipeline_appends_symbol_for_partitioning(monkeypatch: pytest.MonkeyPatch) -> None:
    quote_df = pl.DataFrame(
        {
            "Symbol": ["AAA"],
            "Exchange": ["N"],
            "Bid_Price": [10.0],
            "Offer_Price": [10.1],
            "Time": [94_500_000_000_000],
            "Participant_Timestamp": [94_500_000_000_000],
        },
    )
    monkeypatch.setattr(quote_pipeline, "scan_quote_lazy_frame", lambda _: quote_df.lazy())

    selected_df = quote_pipeline.selected_quote_lazy_frame(
        Path("ignored"),
        symbols=["AAA"],
        exchanges=None,
        columns=("Exchange", "Participant_Timestamp"),
    ).collect()

    assert selected_df.columns == ["Exchange", "Participant_Timestamp", "Symbol"]


def test_trade_writer_writes_audit_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _redirect_output_dirs(monkeypatch, tmp_path)
    raw_path = tmp_path / "raw" / "trade_20251031.txt"
    _write_pipe_csv(
        raw_path,
        schemas.TRADE_RAW_COLUMNS,
        [
            _trade_row(
                Exchange="N",
                Symbol="AAA",
                **{"Participant Timestamp": 94_500_000_000_000},
            ),
            _trade_row(
                Exchange="Q",
                Symbol="BBB",
                **{"Participant Timestamp": 90_000_000_000_000},
            ),
        ],
    )
    monkeypatch.setattr(daily_taq_paths, "trade_input_path", lambda _: raw_path)

    with caplog.at_level(logging.INFO):
        written_paths = write_filtered_trade_parquets(
            ["AAA", "BBB"],
            "20251031",
            columns=("Exchange", "Participant Timestamp"),
        )

    assert list(written_paths) == ["AAA"]
    trade_path = daily_taq_paths.trade_output_path("AAA", "20251031")
    assert written_paths["AAA"] == trade_path
    assert trade_path.exists()
    assert pl.read_parquet(trade_path).columns == ["Exchange", "Participant Timestamp"]

    audit_payload = json.loads(
        daily_taq_paths.trade_audit_path("20251031").read_text(encoding="utf-8"),
    )
    assert audit_payload["written_symbols"] == ["AAA"]
    assert audit_payload["n_written_symbols"] == 1
    assert audit_payload["resolved_output_columns"] == ["Exchange", "Participant Timestamp"]
    assert "trade filtering wrote date=20251031" in caplog.text


def test_quote_writer_writes_requested_shards_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _redirect_output_dirs(monkeypatch, tmp_path)
    shard_paths = {
        "A": tmp_path / "raw" / "quote_A_20251031.txt",
        "M": tmp_path / "raw" / "quote_M_20251031.txt",
    }
    _write_pipe_csv(
        shard_paths["A"],
        schemas.QUOTE_RAW_COLUMNS,
        [
            _quote_row(
                Exchange="N",
                Symbol="AAA",
                Bid_Price=10.0,
                Offer_Price=10.1,
                Time=94_500_000_000_000,
                Participant_Timestamp=94_500_000_000_000,
            ),
            _quote_row(
                Exchange="N",
                Symbol="AXX",
                Bid_Price=10.0,
                Offer_Price=9.9,
                Time=94_500_000_000_000,
                Participant_Timestamp=94_500_000_000_000,
            ),
        ],
    )
    _write_pipe_csv(
        shard_paths["M"],
        schemas.QUOTE_RAW_COLUMNS,
        [
            _quote_row(
                Exchange="Q",
                Symbol="MMM",
                Bid_Price=20.0,
                Offer_Price=20.1,
                Time=94_600_000_000_000,
                Participant_Timestamp=94_600_000_000_000,
            ),
        ],
    )

    def quote_input_paths(_: str, shards: tuple[str, ...] | list[str]) -> list[Path]:
        assert list(shards) == ["A", "M"]
        return [shard_paths[shard] for shard in shards]

    monkeypatch.setattr(daily_taq_paths, "quote_input_paths", quote_input_paths)

    with caplog.at_level(logging.INFO):
        written_paths = write_filtered_quote_parquets(
            ["AAA", "MMM"],
            "20251031",
            columns=("Exchange", "Participant_Timestamp"),
        )

    assert list(written_paths) == ["AAA", "MMM"]
    assert pl.read_parquet(daily_taq_paths.quote_output_path("AAA", "20251031")).columns == [
        "Exchange",
        "Participant_Timestamp",
    ]
    assert pl.read_parquet(daily_taq_paths.quote_output_path("MMM", "20251031")).columns == [
        "Exchange",
        "Participant_Timestamp",
    ]

    audit_payload = json.loads(
        daily_taq_paths.quote_audit_path("20251031").read_text(encoding="utf-8"),
    )
    assert audit_payload["input_paths"] == [str(shard_paths["A"]), str(shard_paths["M"])]
    assert audit_payload["written_symbols"] == ["AAA", "MMM"]
    assert "quote shard processed date=20251031 shard=A" in caplog.text
    assert "quote shard processed date=20251031 shard=M" in caplog.text


def test_trade_cli_build_summary_uses_trade_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inventory, "symbols_with_prefix", lambda *_: ["AAA"])

    def write_trade(*_args: object, **_kwargs: object) -> dict[str, Path]:
        return {"AAA": Path("data/filtered/trade/AAA/trade_20251031.parquet")}

    monkeypatch.setattr(
        trade_cli,
        "write_filtered_trade_parquets",
        write_trade,
    )

    summary = trade_cli.build_trade_summary(target_dates=("20251031",), symbol_prefix="A")

    assert summary == {
        "20251031": {
            "symbol_prefix": "A",
            "symbol_count": 1,
            "columns": ["Exchange", "Participant Timestamp"],
            "audit_path": "data/filtered/trade/_audit/trade_20251031.json",
            "paths": {"AAA": "data/filtered/trade/AAA/trade_20251031.parquet"},
        },
    }


def test_quote_cli_build_summary_uses_quote_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inventory, "symbols_with_prefix", lambda *_: ["AAA"])

    def write_quote(*_args: object, **_kwargs: object) -> dict[str, Path]:
        return {"AAA": Path("data/filtered/quote/AAA/quote_20251031.parquet")}

    monkeypatch.setattr(
        quote_cli,
        "write_filtered_quote_parquets",
        write_quote,
    )

    summary = quote_cli.build_quote_summary(target_dates=("20251031",), symbol_prefix="A")

    assert summary == {
        "20251031": {
            "symbol_prefix": "A",
            "symbol_count": 1,
            "columns": ["Exchange", "Participant_Timestamp"],
            "audit_path": "data/filtered/quote/_audit/quote_20251031.json",
            "paths": {"AAA": "data/filtered/quote/AAA/quote_20251031.parquet"},
        },
    }
