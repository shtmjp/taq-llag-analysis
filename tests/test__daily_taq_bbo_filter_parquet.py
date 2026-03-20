from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import bench_daily_taq_bbo_filter_parquet as bench

if TYPE_CHECKING:
    from pathlib import Path

MARKET_OPEN_SEC = 34_200
MARKET_CLOSE_SEC = 57_600


def _write_pipe_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["|".join(header), *("|".join(row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test__time_to_seconds_expr__open_time() -> None:
    df = pl.DataFrame({"Time": [93_000_000_000_000]})
    out = df.lazy().select(bench.time_to_seconds_expr().alias("time_sec")).collect()
    assert out["time_sec"][0] == MARKET_OPEN_SEC


def test__filtered_bbo_lazy_frame__price_and_market_hours(tmp_path: Path) -> None:
    path = tmp_path / "SPLITS_US_ALL_BBO_A_20200101"
    header = [
        "Time",
        "Participant_Timestamp",
        "Exchange",
        "Symbol",
        "Bid_Price",
        "Bid_Size",
        "Offer_Price",
        "Offer_Size",
        "Quote_Condition",
        "Quote_Cancel_Correction",
    ]
    rows = [
        ["92959000000000", "1", "Q", "A", "1.0", "10", "2.0", "20", "R", ""],  # 09:29:59
        ["93000000000000", "2", "Q", "A", "0.0", "10", "2.0", "20", "R", ""],  # bid=0
        ["93000000000000", "3", "Q", "A", "1.0", "10", "0.0", "20", "R", ""],  # offer=0
        ["93000000000000", "4", "Q", "A", "2.0", "10", "2.0", "20", "R", ""],  # bid==offer
        ["93000000000000", "5", "Q", "A", "2.0", "10", "1.0", "20", "R", ""],  # bid>offer
        ["93000000000000", "6", "Q", "A", "1.0", "10", "2.0", "20", "R", ""],  # ok
        ["160000000000000", "7", "Q", "A", "1.0", "10", "2.0", "20", "R", ""],  # ok (inclusive)
        ["160001000000000", "8", "Q", "A", "1.0", "10", "2.0", "20", "R", ""],  # 16:00:01
        ["END", "20200101", "8", "", "", "", "", "", "", ""],
    ]
    _write_pipe_csv(path, header, rows)

    out = (
        bench.filtered_bbo_lazy_frame(
            path,
            symbols=None,
            exchanges=None,
            use_market_hours=True,
            market_open_sec=MARKET_OPEN_SEC,
            market_close_sec=MARKET_CLOSE_SEC,
            all_columns=False,
            add_time_sec=False,
        )
        .collect()
        .sort("Participant_Timestamp")
    )

    expected_participant_timestamps = [6, 7]
    expected_times = [93_000_000_000_000, 160_000_000_000_000]
    assert out.shape[0] == len(expected_participant_timestamps)
    assert out["Participant_Timestamp"].to_list() == expected_participant_timestamps
    assert out["Time"].to_list() == expected_times


def test__filtered_bbo_lazy_frame__symbols_and_exchanges(tmp_path: Path) -> None:
    path = tmp_path / "SPLITS_US_ALL_BBO_A_20200102"
    header = [
        "Time",
        "Participant_Timestamp",
        "Exchange",
        "Symbol",
        "Bid_Price",
        "Bid_Size",
        "Offer_Price",
        "Offer_Size",
        "Quote_Condition",
        "Quote_Cancel_Correction",
    ]
    rows = [
        ["93000000000000", "1", "Q", "A", "1.0", "10", "2.0", "20", "R", ""],
        ["93000000000000", "2", "T", "B", "1.0", "10", "2.0", "20", "R", ""],
    ]
    _write_pipe_csv(path, header, rows)

    out = (
        bench.filtered_bbo_lazy_frame(
            path,
            symbols=["A"],
            exchanges=["Q"],
            use_market_hours=True,
            market_open_sec=MARKET_OPEN_SEC,
            market_close_sec=MARKET_CLOSE_SEC,
            all_columns=False,
            add_time_sec=False,
        )
        .collect()
        .sort("Participant_Timestamp")
    )
    assert out["Participant_Timestamp"].to_list() == [1]


def test__filtered_bbo_lazy_frame__sink_parquet(tmp_path: Path) -> None:
    path = tmp_path / "SPLITS_US_ALL_BBO_A_20200103"
    header = [
        "Time",
        "Participant_Timestamp",
        "Exchange",
        "Symbol",
        "Bid_Price",
        "Bid_Size",
        "Offer_Price",
        "Offer_Size",
        "Quote_Condition",
        "Quote_Cancel_Correction",
    ]
    rows = [["93000000000000", "1", "Q", "A", "1.0", "10", "2.0", "20", "R", ""]]
    _write_pipe_csv(path, header, rows)

    out_path = tmp_path / "out.parquet"
    bench.filtered_bbo_lazy_frame(
        path,
        symbols=None,
        exchanges=None,
        use_market_hours=True,
        market_open_sec=MARKET_OPEN_SEC,
        market_close_sec=MARKET_CLOSE_SEC,
        all_columns=False,
        add_time_sec=True,
    ).sink_parquet(out_path, statistics=False)

    assert out_path.is_file()
    got = pl.read_parquet(out_path)
    assert got.shape[0] == 1
