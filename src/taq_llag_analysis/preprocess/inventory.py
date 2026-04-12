from __future__ import annotations

import polars as pl

from .daily_taq_paths import master_input_path


def symbols_with_prefix(date_yyyymmdd: str, symbol_prefix: str) -> list[str]:
    """Load symbols whose ticker starts with the requested prefix.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    symbol_prefix
        Prefix to match against the ``Symbol`` column. Pass ``""`` to return
        all symbols.

    Returns
    -------
    list[str]
        Sorted symbol list for the requested date.

    """
    return (
        pl.scan_csv(
            master_input_path(date_yyyymmdd),
            separator="|",
            has_header=True,
            comment_prefix="END",
            schema_overrides={"Symbol": pl.String},
            encoding="utf8-lossy",
        )
        .filter(pl.col("Symbol").str.starts_with(symbol_prefix))
        .select("Symbol")
        .unique()
        .sort("Symbol")
        .collect(engine="streaming")
        .get_column("Symbol")
        .to_list()
    )
