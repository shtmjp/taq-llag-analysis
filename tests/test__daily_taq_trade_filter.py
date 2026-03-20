from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import polars as pl

import daily_taq_trade_filter as trade_filter

if TYPE_CHECKING:
    from pathlib import Path

MARKET_OPEN_SEC = 35_100
MARKET_CLOSE_SEC = 56_700


def _write_pipe_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["|".join(header), *("|".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test__participant_timestamp_sec_expr__market_open() -> None:
    df = pl.DataFrame({"Participant Timestamp": [94_500_000_000_000]})
    out = (
        df.lazy()
        .select(
            trade_filter.participant_timestamp_sec_expr().alias("participant_timestamp_sec"),
        )
        .collect()
    )
    assert out["participant_timestamp_sec"][0] == MARKET_OPEN_SEC


def test__participant_timestamp_time_expr__subsecond_precision() -> None:
    df = pl.DataFrame({"Participant Timestamp": [94_500_123_456_789]})
    out = (
        df.lazy()
        .select(
            trade_filter.participant_timestamp_time_expr().alias("participant_timestamp_time"),
        )
        .collect()
    )
    assert isclose(out["participant_timestamp_time"][0], 35_100.123456789)


def test__filtered_trade_lazy_frame__symbol_exchange_and_market_hours(tmp_path: Path) -> None:
    path = tmp_path / "EQY_US_ALL_TRADE_20200101"
    header = [
        "Time",
        "Exchange",
        "Symbol",
        "Sale Condition",
        "Trade Volume",
        "Trade Price",
        "Trade Stop Stock Indicator",
        "Trade Correction Indicator",
        "Sequence Number",
        "Trade Id",
        "Source of Trade",
        "Trade Reporting Facility",
        "Participant Timestamp",
        "Trade Reporting Facility TRF Timestamp",
        "Trade Through Exempt Indicator",
    ]
    rows = [
        [
            "1",
            "N",
            "AAPL",
            "@",
            "100",
            "190.0",
            "",
            "0",
            "1",
            "10",
            "C",
            "",
            "94459999999999",
            "",
            "0",
        ],
        [
            "2",
            "N",
            "MSFT",
            "@",
            "100",
            "190.0",
            "",
            "0",
            "2",
            "11",
            "C",
            "",
            "94500123456789",
            "",
            "0",
        ],
        [
            "3",
            "P",
            "AAPL",
            "@",
            "100",
            "190.0",
            "",
            "0",
            "3",
            "12",
            "C",
            "",
            "94500200000000",
            "",
            "0",
        ],
        [
            "4",
            "N",
            "AAPL",
            "@",
            "100",
            "190.0",
            "",
            "0",
            "4",
            "13",
            "C",
            "",
            "94500123456789",
            "",
            "0",
        ],
        [
            "5",
            "Q",
            "AAPL",
            "@F",
            "200",
            "191.0",
            "",
            "0",
            "5",
            "14",
            "C",
            "",
            "154500999000000",
            "",
            "0",
        ],
        [
            "6",
            "Q",
            "AAPL",
            "@F",
            "200",
            "191.0",
            "",
            "0",
            "6",
            "15",
            "C",
            "",
            "154501000000000",
            "",
            "0",
        ],
        ["END", "20200101", "6", "", "", "", "", "", "", "", "", "", "", "", ""],
    ]
    _write_pipe_csv(path, header, rows)

    out = (
        trade_filter.filtered_trade_lazy_frame(
            path,
            symbols=["AAPL"],
            exchanges=["N", "Q"],
            market_open_sec=MARKET_OPEN_SEC,
            market_close_sec=MARKET_CLOSE_SEC,
            all_columns=False,
            add_time_columns=True,
        )
        .collect()
        .sort("Participant Timestamp")
    )

    assert out["Exchange"].to_list() == ["N", "Q"]
    assert out["Participant Timestamp"].to_list() == [
        94_500_123_456_789,
        154_500_999_000_000,
    ]
    assert out["participant_timestamp_sec"].to_list() == [35_100, 56_700]


def test__filtered_trade_lazy_frame__sink_parquet(tmp_path: Path) -> None:
    path = tmp_path / "EQY_US_ALL_TRADE_20200102"
    header = [
        "Time",
        "Exchange",
        "Symbol",
        "Sale Condition",
        "Trade Volume",
        "Trade Price",
        "Trade Stop Stock Indicator",
        "Trade Correction Indicator",
        "Sequence Number",
        "Trade Id",
        "Source of Trade",
        "Trade Reporting Facility",
        "Participant Timestamp",
        "Trade Reporting Facility TRF Timestamp",
        "Trade Through Exempt Indicator",
    ]
    rows = [
        [
            "1",
            "N",
            "AAPL",
            "@",
            "100",
            "190.0",
            "",
            "0",
            "1",
            "10",
            "C",
            "",
            "94500123456789",
            "",
            "0",
        ],
    ]
    _write_pipe_csv(path, header, rows)

    out_path = tmp_path / "out.parquet"
    trade_filter.filtered_trade_lazy_frame(
        path,
        symbols=["AAPL"],
        exchanges=["N"],
        market_open_sec=MARKET_OPEN_SEC,
        market_close_sec=MARKET_CLOSE_SEC,
        all_columns=False,
        add_time_columns=True,
    ).sink_parquet(out_path, statistics=False)

    got = pl.read_parquet(out_path)
    assert got.columns == [
        "Exchange",
        "Symbol",
        "Trade Price",
        "Trade Volume",
        "Sale Condition",
        "Participant Timestamp",
        "Trade Correction Indicator",
        "Trade Stop Stock Indicator",
        "participant_timestamp_sec",
        "participant_timestamp_time",
    ]
    assert got.shape == (1, 10)
    assert got["participant_timestamp_sec"].to_list() == [35_100]
    assert isclose(got["participant_timestamp_time"][0], 35_100.123456789)
