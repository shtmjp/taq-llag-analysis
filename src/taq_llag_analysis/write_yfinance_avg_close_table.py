from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

START = "2025-09-01"
END = "2025-10-01"
DEFAULT_OUT = Path("data/derived/yfinance_avg_close_sep2025.csv")


def normalize_for_yahoo(symbol: str) -> str:
    """Normalize one symbol for Yahoo Finance.

    Parameters
    ----------
    symbol
        Raw symbol string.

    Returns
    -------
    str
        Yahoo-style symbol string.

    """
    normalized = str(symbol).strip().upper()
    return normalized.replace(".", "-").replace("/", "-")


def read_symbols(path: str | Path, column: str | None = None) -> pd.DataFrame:
    """Load symbols from a text or tabular file.

    Parameters
    ----------
    path
        Input path. Supported formats are `.txt`, `.csv`, `.tsv`, `.xlsx`, and `.xls`.
    column
        Column name to read for tabular inputs. When omitted, the table must have
        exactly one column.

    Returns
    -------
    pandas.DataFrame
        Two-column table with `symbol_original` and `symbol_yahoo`.

    """
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".txt":
        raw_symbols = input_path.read_text(encoding="utf-8").splitlines()
        symbol_table = pd.DataFrame({"symbol_original": raw_symbols})
    elif suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        input_df = pd.read_csv(input_path, sep=separator)
        selected_column = column
        if selected_column is None:
            if len(input_df.columns) != 1:
                message = (
                    f"Column name is ambiguous. Available columns: {list(input_df.columns)}. "
                    "Pass --column."
                )
                raise ValueError(message)
            selected_column = str(input_df.columns[0])
        symbol_table = input_df[[selected_column]].rename(
            columns={selected_column: "symbol_original"},
        )
    elif suffix in {".xlsx", ".xls"}:
        input_df = pd.read_excel(input_path)
        selected_column = column
        if selected_column is None:
            if len(input_df.columns) != 1:
                message = (
                    f"Column name is ambiguous. Available columns: {list(input_df.columns)}. "
                    "Pass --column."
                )
                raise ValueError(message)
            selected_column = str(input_df.columns[0])
        symbol_table = input_df[[selected_column]].rename(
            columns={selected_column: "symbol_original"},
        )
    else:
        message = "Supported input formats: .txt, .csv, .tsv, .xlsx, .xls"
        raise ValueError(message)

    symbol_table = symbol_table.dropna(subset=["symbol_original"]).copy()
    symbol_table["symbol_original"] = symbol_table["symbol_original"].astype(str).str.strip()
    symbol_table = symbol_table.loc[symbol_table["symbol_original"].ne("")].copy()
    symbol_table = symbol_table.loc[symbol_table["symbol_original"].str.lower().ne("nan")].copy()
    symbol_table["symbol_yahoo"] = symbol_table["symbol_original"].map(normalize_for_yahoo)
    return symbol_table.drop_duplicates(subset=["symbol_original"]).reset_index(
        drop=True,
    )


def batched(xs: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield a list in contiguous batches.

    Parameters
    ----------
    xs
        Input symbol list.
    batch_size
        Number of symbols per batch.

    Yields
    ------
    list[str]
        One symbol batch.

    """
    for start_index in range(0, len(xs), batch_size):
        yield xs[start_index : start_index + batch_size]


def download_close_batch(
    symbols: list[str],
    start: str,
    end: str,
    max_retries: int = 5,
    base_sleep: float = 2.0,
) -> pd.DataFrame:
    """Download one batch of daily close prices from Yahoo Finance.

    Parameters
    ----------
    symbols
        Yahoo-normalized symbols.
    start
        Inclusive start date in `YYYY-MM-DD` format.
    end
        Exclusive end date in `YYYY-MM-DD` format.
    max_retries
        Maximum number of attempts.
    base_sleep
        Base seconds for exponential backoff.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date with one close-price column per symbol.

    """
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=False,
                progress=False,
                multi_level_index=True,
                timeout=30,
            )

            if data is None or data.empty:
                return pd.DataFrame()

            if not isinstance(data.columns, pd.MultiIndex):
                data = pd.concat({symbols[0]: data}, axis=1)

            first_level = set(data.columns.get_level_values(0))
            close_frames = [
                data[symbol][["Close"]].rename(columns={"Close": symbol})
                for symbol in symbols
                if symbol in first_level
            ]

            if not close_frames:
                return pd.DataFrame(index=pd.Index([], name="Date"))

            close = pd.concat(close_frames, axis=1)
            index = pd.to_datetime(close.index)
            if getattr(index, "tz", None) is not None:
                index = index.tz_localize(None)
            close.index = index
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            sleep_seconds = base_sleep * (2**attempt)
            sys.stdout.write(
                f"[retry {attempt + 1}/{max_retries}] "
                f"batch starting at {symbols[0]!r} failed: {exc!r}. "
                f"sleep {sleep_seconds:.1f}s\n",
            )
            time.sleep(sleep_seconds)
        else:
            return close.sort_index()

    message = f"Failed batch: {symbols[:5]} ..."
    raise RuntimeError(message) from last_exc


def make_avg_close_table(
    symbol_table: pd.DataFrame,
    start: str = START,
    end: str = END,
    batch_size: int = 50,
    sleep_between_batches: float = 1.0,
    download_fn: Callable[[list[str], str, str], pd.DataFrame] = download_close_batch,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-symbol average close and daily close tables.

    Parameters
    ----------
    symbol_table
        DataFrame with `symbol_original` and `symbol_yahoo`.
    start
        Inclusive start date.
    end
        Exclusive end date.
    batch_size
        Number of Yahoo symbols per request batch.
    sleep_between_batches
        Seconds to wait between batches.
    download_fn
        Batch download function, mainly for testing.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Summary table and daily close panel.

    """
    yahoo_symbols = symbol_table["symbol_yahoo"].drop_duplicates().tolist()

    daily_close_parts: list[pd.DataFrame] = []
    failed_symbols: list[str] = []

    for batch in batched(yahoo_symbols, batch_size):
        close = download_fn(batch, start, end)

        if close.empty:
            failed_symbols.extend(batch)
        else:
            daily_close_parts.append(close)
            returned = set(close.columns.astype(str))
            failed_symbols.extend([symbol for symbol in batch if symbol not in returned])

        time.sleep(sleep_between_batches)

    if daily_close_parts:
        daily_close = pd.concat(daily_close_parts, axis=1)
        daily_close = daily_close.loc[:, ~daily_close.columns.duplicated()]
        daily_close = daily_close.sort_index()
    else:
        daily_close = pd.DataFrame()

    if daily_close.empty:
        summary = symbol_table.copy()
        summary["avg_close"] = pd.NA
        summary["n_obs"] = 0
        summary["status"] = "missing"
        return summary, daily_close

    mean_close = daily_close.mean(axis=0, skipna=True)
    n_obs = daily_close.notna().sum(axis=0)
    per_yahoo = pd.DataFrame(
        {
            "symbol_yahoo": mean_close.index.astype(str),
            "avg_close": mean_close.to_numpy(),
            "n_obs": n_obs.reindex(mean_close.index).to_numpy(),
        },
    )

    summary = symbol_table.merge(per_yahoo, on="symbol_yahoo", how="left")
    summary["status"] = "ok"
    summary.loc[summary["n_obs"].isna() | (summary["n_obs"] == 0), "status"] = "missing"

    failed_set = set(failed_symbols)
    summary.loc[
        summary["symbol_yahoo"].isin(failed_set) & summary["avg_close"].isna(),
        "status",
    ] = "failed_or_unmapped"

    summary = summary[
        ["symbol_original", "symbol_yahoo", "avg_close", "n_obs", "status"]
    ].sort_values(["status", "symbol_original"], ascending=[True, True])
    return summary.reset_index(drop=True), daily_close


def main() -> int:
    """Run the Yahoo Finance average-close export workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        required=True,
        help="Path to txt/csv/tsv/xlsx/xls file",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Column name if input is tabular",
    )
    parser.add_argument(
        "--start",
        default=START,
        help="Inclusive start date, e.g. 2025-09-01",
    )
    parser.add_argument(
        "--end",
        default=END,
        help="Exclusive end date, e.g. 2025-10-01",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between batches",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--daily-out",
        default=None,
        help="Optional path for daily close panel",
    )
    args = parser.parse_args()

    symbol_table = read_symbols(args.symbols, column=args.column)
    summary, daily_close = make_avg_close_table(
        symbol_table=symbol_table,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    if args.daily_out:
        daily_out_path = Path(args.daily_out)
        daily_out_path.parent.mkdir(parents=True, exist_ok=True)
        daily_close.to_csv(daily_out_path, index=True)

    n_ok = int((summary["status"] == "ok").sum())
    n_total = len(summary)
    sys.stdout.write(f"saved: {out_path}\n")
    if args.daily_out:
        sys.stdout.write(f"saved: {Path(args.daily_out)}\n")
    sys.stdout.write(f"ok: {n_ok}/{n_total}\n")
    sys.stdout.write(f"{summary.head(10).to_string(index=False)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
