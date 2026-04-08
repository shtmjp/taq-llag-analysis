from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from write_yfinance_avg_close_table import make_avg_close_table, read_symbols

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_AVG_CLOSE_AAA = 11.0
EXPECTED_N_OBS_AAA = 2
EXPECTED_N_OBS_CCC = 0


def test_read_symbols_csv_normalizes_and_deduplicates(tmp_path: Path) -> None:
    symbols_path = tmp_path / "symbols.csv"
    pd.DataFrame(
        {
            "ticker": ["brk.b", " BRK.B ", "msft", "", "nan"],
        },
    ).to_csv(symbols_path, index=False)

    symbol_table = read_symbols(symbols_path, column="ticker")

    assert symbol_table.to_dict(orient="records") == [
        {
            "symbol_original": "brk.b",
            "symbol_yahoo": "BRK-B",
        },
        {
            "symbol_original": "BRK.B",
            "symbol_yahoo": "BRK-B",
        },
        {
            "symbol_original": "msft",
            "symbol_yahoo": "MSFT",
        },
    ]


def test_make_avg_close_table_classifies_symbol_statuses() -> None:
    symbol_table = pd.DataFrame(
        {
            "symbol_original": ["AAA", "BBB", "CCC"],
            "symbol_yahoo": ["AAA", "BBB", "CCC"],
        },
    )

    def fake_download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
        assert start == "2025-09-01"
        assert end == "2025-10-01"
        if symbols == ["AAA", "BBB"]:
            return pd.DataFrame(
                {
                    "AAA": [10.0, 12.0],
                },
                index=pd.to_datetime(["2025-09-02", "2025-09-03"]),
            )
        if symbols == ["CCC"]:
            return pd.DataFrame(
                {
                    "CCC": [pd.NA, pd.NA],
                },
                index=pd.to_datetime(["2025-09-02", "2025-09-03"]),
            )
        raise AssertionError(symbols)

    summary, daily_close = make_avg_close_table(
        symbol_table=symbol_table,
        batch_size=2,
        sleep_between_batches=0.0,
        download_fn=fake_download,
    )

    assert list(daily_close.columns) == ["AAA", "CCC"]
    aaa_row = summary.loc[summary["symbol_original"] == "AAA"].iloc[0]
    assert aaa_row["avg_close"] == EXPECTED_AVG_CLOSE_AAA
    assert aaa_row["n_obs"] == EXPECTED_N_OBS_AAA
    assert aaa_row["status"] == "ok"

    bbb_row = summary.loc[summary["symbol_original"] == "BBB"].iloc[0]
    assert pd.isna(bbb_row["avg_close"])
    assert pd.isna(bbb_row["n_obs"])
    assert bbb_row["status"] == "failed_or_unmapped"

    ccc_row = summary.loc[summary["symbol_original"] == "CCC"].iloc[0]
    assert pd.isna(ccc_row["avg_close"])
    assert ccc_row["n_obs"] == EXPECTED_N_OBS_CCC
    assert ccc_row["status"] == "missing"
