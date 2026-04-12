from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


RAW_ROOT_DIR = Path("/Users/shiotanitenshou/Research/datasets/dailytaq2")
TRADE_OUTPUT_DIR = Path("data/filtered/trade")
QUOTE_OUTPUT_DIR = Path("data/filtered/quote")
TRADE_AUDIT_DIR = TRADE_OUTPUT_DIR / "_audit"
QUOTE_AUDIT_DIR = QUOTE_OUTPUT_DIR / "_audit"
MARKET_OPEN_SEC = 35_100
MARKET_CLOSE_SEC = 56_700

MASTER_RAW_COLUMNS: tuple[str, ...] = (
    "Symbol",
    "Security_Description",
    "CUSIP",
    "Security_Type",
    "SIP_Symbol",
    "Old_Symbol",
    "Test_Symbol_Flag",
    "Listed_Exchange",
    "Tape",
    "Unit_Of_Trade",
    "Round_Lot",
    "NYSE_Industry_Code",
    "Shares_Outstanding",
    "Halt_Delay_Reason",
    "Specialist_Clearing_Agent",
    "Specialist_Clearing_Number",
    "Specialist_Post_Number",
    "Specialist_Panel",
    "TradedOnNYSEMKT",
    "TradedOnNASDAQBX",
    "TradedOnNSX",
    "TradedOnFINRA",
    "TradedOnISE",
    "TradedOnEdgeA",
    "TradedOnEdgeX",
    "TradedOnNYSETexas",
    "TradedOnNYSE",
    "TradedOnArca",
    "TradedOnNasdaq",
    "TradedOnCBOE",
    "TradedOnPSX",
    "TradedOnBATSY",
    "TradedOnBATS",
    "TradedOnIEX",
    "Tick_Pilot_Indicator",
    "Effective_Date",
    "TradedOnLTSE",
    "TradedOnMEMX",
    "TradedOnMIAX",
    "TradedOn24X",
)

TRADE_RAW_COLUMNS: tuple[str, ...] = (
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
)

QUOTE_RAW_COLUMNS: tuple[str, ...] = (
    "Time",
    "Exchange",
    "Symbol",
    "Bid_Price",
    "Bid_Size",
    "Offer_Price",
    "Offer_Size",
    "Quote_Condition",
    "Sequence_Number",
    "National_BBO_Ind",
    "FINRA_BBO_Indicator",
    "FINRA_ADF_MPID_Indicator",
    "Quote_Cancel_Correction",
    "Source_Of_Quote",
    "Retail_Interest_Indicator",
    "Short_Sale_Restriction_Indicator",
    "LULD_BBO_Indicator",
    "SIP_Generated_Message_Identifier",
    "National_BBO_LULD_Indicator",
    "Participant_Timestamp",
    "FINRA_ADF_Timestamp",
    "FINRA_ADF_Market_Participant_Quote_Indicator",
    "Security_Status_Indicator",
)

TRADE_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "Time": pl.Int64,
    "Exchange": pl.String,
    "Symbol": pl.String,
    "Sale Condition": pl.String,
    "Trade Volume": pl.Int64,
    "Trade Price": pl.Float64,
    "Trade Stop Stock Indicator": pl.String,
    "Trade Correction Indicator": pl.Int64,
    "Sequence Number": pl.Int64,
    "Trade Id": pl.Int64,
    "Source of Trade": pl.String,
    "Trade Reporting Facility": pl.String,
    "Participant Timestamp": pl.Int64,
    "Trade Reporting Facility TRF Timestamp": pl.String,
    "Trade Through Exempt Indicator": pl.Int64,
}

QUOTE_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "Time": pl.Int64,
    "Exchange": pl.String,
    "Symbol": pl.String,
    "Bid_Price": pl.Float64,
    "Bid_Size": pl.Int64,
    "Offer_Price": pl.Float64,
    "Offer_Size": pl.Int64,
    "Quote_Condition": pl.String,
    "Sequence_Number": pl.Int64,
    "National_BBO_Ind": pl.String,
    "FINRA_BBO_Indicator": pl.String,
    "FINRA_ADF_MPID_Indicator": pl.String,
    "Quote_Cancel_Correction": pl.String,
    "Source_Of_Quote": pl.String,
    "Retail_Interest_Indicator": pl.String,
    "Short_Sale_Restriction_Indicator": pl.String,
    "LULD_BBO_Indicator": pl.String,
    "SIP_Generated_Message_Identifier": pl.String,
    "National_BBO_LULD_Indicator": pl.String,
    "Participant_Timestamp": pl.Int64,
    "FINRA_ADF_Timestamp": pl.Int64,
    "FINRA_ADF_Market_Participant_Quote_Indicator": pl.String,
    "Security_Status_Indicator": pl.String,
}


def master_input_path(date_yyyymmdd: str) -> Path:
    """Resolve the Daily TAQ master file path for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.

    Returns
    -------
    pathlib.Path
        Gzipped master-file path for the requested date.

    """
    year = date_yyyymmdd[:4]
    year_month = date_yyyymmdd[:6]
    return (
        RAW_ROOT_DIR
        / f"EQY_US_ALL_REF_MASTER_{year}"
        / f"EQY_US_ALL_REF_MASTER_{year_month}"
        / f"EQY_US_ALL_REF_MASTER_{date_yyyymmdd}.gz"
    )


def trade_input_path(date_yyyymmdd: str) -> Path:
    """Resolve the Daily TAQ trade file path for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.

    Returns
    -------
    pathlib.Path
        Gzipped trade-file path for the requested date.

    """
    year = date_yyyymmdd[:4]
    year_month = date_yyyymmdd[:6]
    return (
        RAW_ROOT_DIR
        / f"EQY_US_ALL_TRADE_{year}"
        / f"EQY_US_ALL_TRADE_{year_month}"
        / f"EQY_US_ALL_TRADE_{date_yyyymmdd}"
        / f"EQY_US_ALL_TRADE_{date_yyyymmdd}.gz"
    )


def quote_input_paths(date_yyyymmdd: str, shards: Sequence[str]) -> list[Path]:
    """Resolve Daily TAQ quote shard paths for one date.

    Parameters
    ----------
    date_yyyymmdd
        Trading date formatted as ``YYYYMMDD``.
    shards
        Quote shard tokens such as ``"A"`` or ``"M"``.

    Returns
    -------
    list[pathlib.Path]
        Gzipped BBO shard paths for the requested date.

    """
    year = date_yyyymmdd[:4]
    year_month = date_yyyymmdd[:6]
    base_dir = (
        RAW_ROOT_DIR
        / f"SPLITS_US_ALL_BBO_{year}"
        / f"SPLITS_US_ALL_BBO_{year_month}"
        / f"SPLITS_US_ALL_BBO_{date_yyyymmdd}"
    )
    return [
        base_dir / f"SPLITS_US_ALL_BBO_{shard}_{date_yyyymmdd}.gz" for shard in sorted(set(shards))
    ]


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


def trade_audit_path(date_yyyymmdd: str) -> Path:
    """Return the audit JSON path for one filtered trade date."""
    return TRADE_AUDIT_DIR / f"trade_{date_yyyymmdd}.json"


def quote_audit_path(date_yyyymmdd: str) -> Path:
    """Return the audit JSON path for one filtered quote date."""
    return QUOTE_AUDIT_DIR / f"quote_{date_yyyymmdd}.json"


def hms_integer_seconds_expr(time_col: str) -> pl.Expr:
    """Convert a Daily TAQ ``HHMMSSxxxxxxxxx`` integer column to seconds."""
    time_value = pl.col(time_col)
    hh = time_value // 10_000_000_000_000
    mm = (time_value // 100_000_000_000) % 100
    ss = (time_value // 1_000_000_000) % 100
    return hh * 3600 + mm * 60 + ss


def resolve_output_columns(
    requested_columns: Sequence[str] | None,
    available_columns: Sequence[str],
) -> list[str]:
    """Resolve the raw columns that should be written to parquet."""
    if requested_columns is None:
        return list(available_columns)
    return list(requested_columns)


def append_if_missing(columns: Sequence[str], extra_column: str) -> list[str]:
    """Return ``columns`` with ``extra_column`` appended when needed."""
    resolved_columns = list(columns)
    if extra_column not in resolved_columns:
        resolved_columns.append(extra_column)
    return resolved_columns


def utc_now_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 ``Z`` form."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_audit_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Write an audit JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def stream_lazy_frames_to_symbol_parquets(
    lazy_frames: Sequence[tuple[str, pl.LazyFrame]],
    *,
    output_columns: Sequence[str],
    output_path_for_symbol: Callable[[str], Path],
) -> tuple[dict[str, Path], int, float, float]:
    """Stream filtered lazy frames into per-symbol parquet files.

    Parameters
    ----------
    lazy_frames
        Sequence of labeled lazy frames. The labels are used only to make
        temporary chunk filenames unique.
    output_columns
        Columns to keep in the final parquet files.
    output_path_for_symbol
        Function mapping one symbol to its final parquet path.

    Returns
    -------
    tuple[dict[str, pathlib.Path], int, float, float]
        Final parquet paths by symbol, total filtered row count, chunking
        elapsed seconds, and merge elapsed seconds.

    """
    chunk_paths_by_symbol: dict[str, list[Path]] = {}
    n_filtered_rows_total = 0

    chunking_start = time.perf_counter()
    with TemporaryDirectory(prefix="daily_taq_symbol_chunks_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for source_label, lazy_frame in lazy_frames:
            for batch_index, batch_df in enumerate(
                lazy_frame.collect_batches(engine="streaming"),
            ):
                if batch_df.height == 0:
                    continue

                n_filtered_rows_total += batch_df.height
                for symbol_df in batch_df.partition_by("Symbol", maintain_order=False):
                    symbol = str(symbol_df.get_column("Symbol")[0])
                    chunk_path = temp_dir / symbol / f"{source_label}_{batch_index:06d}.parquet"
                    chunk_path.parent.mkdir(parents=True, exist_ok=True)
                    symbol_df.select(output_columns).write_parquet(
                        chunk_path,
                        statistics=False,
                    )
                    chunk_paths_by_symbol.setdefault(symbol, []).append(chunk_path)
        chunking_elapsed_sec = time.perf_counter() - chunking_start

        merge_start = time.perf_counter()
        written_paths: dict[str, Path] = {}
        for symbol, chunk_paths in chunk_paths_by_symbol.items():
            output_path = output_path_for_symbol(symbol)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            pl.scan_parquet([str(path) for path in chunk_paths]).sink_parquet(
                output_path,
                statistics=False,
            )
            written_paths[symbol] = output_path
        merge_elapsed_sec = time.perf_counter() - merge_start

    return written_paths, n_filtered_rows_total, chunking_elapsed_sec, merge_elapsed_sec
