from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from daily_taq_quote_filter import filtered_quote_lazy_frame

if TYPE_CHECKING:
    from collections.abc import Sequence


QUOTE_INPUT_DIR = Path("data/dailyTAQ/BBO")
QUOTE_OUTPUT_DIR = Path("data/filtered/quote")
MARKET_OPEN_SEC = 35_100
MARKET_CLOSE_SEC = 56_700


def _quote_input_path(date_yyyymmdd: str, shard: str) -> Path:
    plain_path = QUOTE_INPUT_DIR / f"SPLITS_US_ALL_BBO_{shard}_{date_yyyymmdd}"
    gz_path = plain_path.with_suffix(".gz")
    return plain_path if plain_path.exists() else gz_path


def write_filtered_quote_parquets(
    symbols: Sequence[str],
    date_yyyymmdd: str,
    shard: str,
) -> dict[str, Path]:
    """Write filtered Daily TAQ quote parquet files for one date and shard.

    Parameters
    ----------
    symbols
        Symbols to keep.
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    shard
        BBO shard token used in ``SPLITS_US_ALL_BBO_{shard}_{date}``.

    Returns
    -------
    dict[str, pathlib.Path]
        Output parquet path for each symbol that has at least one filtered
        quote row.

    """
    if not symbols:
        return {}

    input_path = _quote_input_path(date_yyyymmdd, shard)
    filtered_df = filtered_quote_lazy_frame(
        input_path,
        symbols=symbols,
        exchanges=None,
        use_market_hours=True,
        market_open_sec=MARKET_OPEN_SEC,
        market_close_sec=MARKET_CLOSE_SEC,
        all_columns=True,
        add_time_sec=False,
    ).collect(engine="streaming")

    written_paths: dict[str, Path] = {}
    for symbol_df in filtered_df.partition_by("Symbol", maintain_order=False):
        symbol = symbol_df.get_column("Symbol")[0]
        output_path = QUOTE_OUTPUT_DIR / symbol / f"quote_{date_yyyymmdd}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        symbol_df.write_parquet(output_path, statistics=False)
        written_paths[symbol] = output_path

    return {symbol: written_paths[symbol] for symbol in symbols if symbol in written_paths}
