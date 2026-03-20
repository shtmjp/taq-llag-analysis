from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_INPUT_DIR = Path("data/dailyTAQ")
DEFAULT_OUTPUT_DIR = Path("data/derived/daily_taq_bbo_parquet")

DEFAULT_OUTPUT_COLUMNS: tuple[str, ...] = (
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
)

DEFAULT_SCHEMA_OVERRIDES: dict[str, pl.DataType] = {
    "Time": pl.Int64,
    "Participant_Timestamp": pl.Int64,
    "Exchange": pl.String,
    "Symbol": pl.String,
    "Bid_Price": pl.Float64,
    "Bid_Size": pl.Int64,
    "Offer_Price": pl.Float64,
    "Offer_Size": pl.Int64,
    "Quote_Condition": pl.String,
    "Quote_Cancel_Correction": pl.String,
}


@dataclass(frozen=True, slots=True)
class BenchResult:
    """Benchmark result for a single input -> parquet write.

    Attributes
    ----------
    input_path
        Source Daily TAQ file.
    output_path
        Destination parquet file.
    input_bytes
        Source file size in bytes.
    output_bytes
        Output parquet size in bytes.
    wall_seconds
        Wall-clock elapsed time for filtering + parquet write.
    cpu_seconds
        Process CPU time for filtering + parquet write.

    """

    input_path: Path
    output_path: Path
    input_bytes: int
    output_bytes: int
    wall_seconds: float
    cpu_seconds: float

    @property
    def throughput_gib_per_s(self) -> float:
        """Compute input throughput in GiB/s (based on wall time).

        Returns
        -------
        float
            Input throughput in GiB/s.

        """
        gib = 1024**3
        return (self.input_bytes / gib) / self.wall_seconds if self.wall_seconds else float("inf")


@dataclass(frozen=True, slots=True)
class _SplitBenchResult:
    input_path: Path
    output_path: Path
    input_bytes: int
    output_bytes: int
    read_wall_seconds: float
    read_cpu_seconds: float
    sink_wall_seconds: float
    sink_cpu_seconds: float
    n_rows: int

    @property
    def approx_output_wall_seconds(self) -> float:
        return self.sink_wall_seconds - self.read_wall_seconds

    @property
    def approx_output_cpu_seconds(self) -> float:
        return self.sink_cpu_seconds - self.read_cpu_seconds

    @property
    def read_throughput_gib_per_s(self) -> float:
        gib = 1024**3
        return (
            (self.input_bytes / gib) / self.read_wall_seconds
            if self.read_wall_seconds
            else float("inf")
        )

    @property
    def sink_throughput_gib_per_s(self) -> float:
        gib = 1024**3
        return (
            (self.input_bytes / gib) / self.sink_wall_seconds
            if self.sink_wall_seconds
            else float("inf")
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Daily TAQ BBO (SPLITS_US_ALL_BBO_*) with Polars LazyFrame and write Parquet, "
            "measuring end-to-end performance."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefer", choices=("gz", "plain"), default="gz")

    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--exchanges", nargs="+", default=None)

    parser.add_argument("--market-open-sec", type=int, default=34_200)
    parser.add_argument("--market-close-sec", type=int, default=57_600)
    parser.add_argument("--no-market-hours", action="store_true")

    parser.add_argument("--all-columns", action="store_true")
    parser.add_argument("--add-time-sec", action="store_true")

    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--statistics", action="store_true")
    parser.add_argument("--row-group-size", type=int, default=None)

    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--split-timing", action="store_true")

    return parser.parse_args(argv)


def time_to_seconds_expr(time_col: str = "Time") -> pl.Expr:
    """Convert Daily TAQ integer time to seconds since midnight.

    Notes
    -----
    Daily TAQ encodes `Time` as an integer in the form:

    - `HHMMSSxxxxxxxxx` (typically 9 sub-second digits)

    The conversion here follows the integer arithmetic used in the referenced PDF example.

    Parameters
    ----------
    time_col
        Column name for the integer time field.

    Returns
    -------
    polars.Expr
        Expression that evaluates to seconds since midnight.

    """
    t = pl.col(time_col)
    hh = t // 10_000_000_000_000
    mm = (t // 100_000_000_000) % 100
    ss = (t // 1_000_000_000) % 100
    return hh * 3600 + mm * 60 + ss


def _select_preferred_bbo_files(input_dir: Path, *, prefer: str) -> list[Path]:
    candidates = sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.name.startswith("SPLITS_US_ALL_BBO_")),
        key=lambda p: p.name,
    )

    by_base: dict[str, dict[str, Path]] = {}
    for path in candidates:
        if path.suffix == ".gz":
            base = path.name.removesuffix(".gz")
            by_base.setdefault(base, {})["gz"] = path
        else:
            by_base.setdefault(path.name, {})["plain"] = path

    chosen: list[Path] = []
    for base in sorted(by_base):
        entry = by_base[base]
        if prefer == "gz":
            chosen.append(entry.get("gz", entry["plain"]))
        else:
            chosen.append(entry.get("plain", entry["gz"]))
    return chosen


def filtered_bbo_lazy_frame(
    input_path: Path,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    use_market_hours: bool,
    market_open_sec: int,
    market_close_sec: int,
    all_columns: bool,
    add_time_sec: bool,
) -> pl.LazyFrame:
    """Build a filtered LazyFrame for Daily TAQ BBO data.

    Filtering rules (PDF-aligned):

    - Drop zero bid/offer prices.
    - Keep strictly positive spreads (`Bid_Price < Offer_Price`).
    - Optionally restrict to regular market hours (default in CLI): 09:30--16:00.

    Parameters
    ----------
    input_path
        Path to `SPLITS_US_ALL_BBO_*` (plain text or `.gz`).
    symbols
        Optional symbol whitelist.
    exchanges
        Optional exchange whitelist.
    use_market_hours
        If True, apply the regular-hours time window.
    market_open_sec
        Open time (seconds since midnight).
    market_close_sec
        Close time (seconds since midnight).
    all_columns
        If True, keep all input columns.
    add_time_sec
        If True, include `time_sec` in the output.

    Returns
    -------
    polars.LazyFrame
        Filtered lazy query.

    """
    schema_overrides: dict[str, pl.DataType] | None = DEFAULT_SCHEMA_OVERRIDES

    lf = pl.scan_csv(
        input_path,
        separator="|",
        has_header=True,
        comment_prefix="END",
        schema_overrides=schema_overrides,
    )

    if symbols:
        lf = lf.filter(pl.col("Symbol").is_in(symbols))
    if exchanges:
        lf = lf.filter(pl.col("Exchange").is_in(exchanges))

    lf = lf.filter(
        (pl.col("Bid_Price") > 0)
        & (pl.col("Offer_Price") > 0)
        & (pl.col("Bid_Price") < pl.col("Offer_Price")),
    )

    time_sec = time_to_seconds_expr()
    if use_market_hours:
        lf = lf.filter((time_sec >= market_open_sec) & (time_sec <= market_close_sec))

    if add_time_sec:
        lf = lf.with_columns(time_sec.alias("time_sec"))

    if not all_columns:
        out_cols = list(DEFAULT_OUTPUT_COLUMNS)
        if add_time_sec:
            out_cols.append("time_sec")
        lf = lf.select(out_cols)

    return lf


def _sink_parquet(
    lf: pl.LazyFrame,
    output_path: Path,
    *,
    compression: str,
    statistics: bool,
    row_group_size: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    sink_kwargs: dict[str, object] = {
        "compression": compression,
        "statistics": statistics,
    }
    if row_group_size is not None:
        sink_kwargs["row_group_size"] = row_group_size

    lf.sink_parquet(output_path, **sink_kwargs)


def _bench_one(
    input_path: Path,
    output_path: Path,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    use_market_hours: bool,
    market_open_sec: int,
    market_close_sec: int,
    all_columns: bool,
    add_time_sec: bool,
    compression: str,
    statistics: bool,
    row_group_size: int | None,
) -> BenchResult:
    input_bytes = input_path.stat().st_size

    lf = filtered_bbo_lazy_frame(
        input_path,
        symbols=symbols,
        exchanges=exchanges,
        use_market_hours=use_market_hours,
        market_open_sec=market_open_sec,
        market_close_sec=market_close_sec,
        all_columns=all_columns,
        add_time_sec=add_time_sec,
    )

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    _sink_parquet(
        lf,
        output_path,
        compression=compression,
        statistics=statistics,
        row_group_size=row_group_size,
    )
    cpu_end = time.process_time()
    wall_end = time.perf_counter()

    output_bytes = output_path.stat().st_size
    return BenchResult(
        input_path=input_path,
        output_path=output_path,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        wall_seconds=wall_end - wall_start,
        cpu_seconds=cpu_end - cpu_start,
    )


def _bench_split(
    input_path: Path,
    output_path: Path,
    *,
    symbols: Sequence[str] | None,
    exchanges: Sequence[str] | None,
    use_market_hours: bool,
    market_open_sec: int,
    market_close_sec: int,
    all_columns: bool,
    add_time_sec: bool,
    compression: str,
    statistics: bool,
    row_group_size: int | None,
) -> _SplitBenchResult:
    input_bytes = input_path.stat().st_size

    lf = filtered_bbo_lazy_frame(
        input_path,
        symbols=symbols,
        exchanges=exchanges,
        use_market_hours=use_market_hours,
        market_open_sec=market_open_sec,
        market_close_sec=market_close_sec,
        all_columns=all_columns,
        add_time_sec=add_time_sec,
    )

    read_wall_start = time.perf_counter()
    read_cpu_start = time.process_time()
    n_rows = int(lf.select(pl.len()).collect()[0, 0])
    read_cpu_end = time.process_time()
    read_wall_end = time.perf_counter()

    sink_wall_start = time.perf_counter()
    sink_cpu_start = time.process_time()
    _sink_parquet(
        lf,
        output_path,
        compression=compression,
        statistics=statistics,
        row_group_size=row_group_size,
    )
    sink_cpu_end = time.process_time()
    sink_wall_end = time.perf_counter()

    output_bytes = output_path.stat().st_size
    return _SplitBenchResult(
        input_path=input_path,
        output_path=output_path,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        read_wall_seconds=read_wall_end - read_wall_start,
        read_cpu_seconds=read_cpu_end - read_cpu_start,
        sink_wall_seconds=sink_wall_end - sink_wall_start,
        sink_cpu_seconds=sink_cpu_end - sink_cpu_start,
        n_rows=n_rows,
    )


def _format_bytes_gib(num_bytes: int) -> str:
    gib = 1024**3
    return f"{num_bytes / gib:.3f} GiB"


def _write_line(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entrypoint.

    Parameters
    ----------
    argv
        Command-line arguments, excluding the program name.

    Returns
    -------
    int
        Process exit code.

    """
    args = _parse_args(argv)

    input_paths = _select_preferred_bbo_files(args.input_dir, prefer=args.prefer)
    if not input_paths:
        _write_line(f"No SPLITS_US_ALL_BBO_* files found under: {args.input_dir}")
        return 2

    runs = int(args.runs)
    use_market_hours = not bool(args.no_market_hours)

    if args.split_timing:
        _write_line(
            "input_path\toutput_path\tinput_size\toutput_size\tread_wall_s\tread_cpu_s\t"
            "sink_wall_s\tsink_cpu_s\tapprox_output_wall_s\tapprox_output_cpu_s\t"
            "read_throughput_GiB/s\tsink_throughput_GiB/s\tn_rows",
        )
    else:
        _write_line(
            "input_path\toutput_path\tinput_size\toutput_size\twall_s\tcpu_s\tthroughput_GiB/s",
        )
    for input_path in input_paths:
        base = input_path.name.removesuffix(".gz")
        for run_idx in range(1, runs + 1):
            out_name = (
                f"{base}.filtered.parquet" if runs == 1 else f"{base}.run{run_idx}.filtered.parquet"
            )
            output_path = args.output_dir / out_name
            if args.split_timing:
                result = _bench_split(
                    input_path,
                    output_path,
                    symbols=args.symbols,
                    exchanges=args.exchanges,
                    use_market_hours=use_market_hours,
                    market_open_sec=int(args.market_open_sec),
                    market_close_sec=int(args.market_close_sec),
                    all_columns=bool(args.all_columns),
                    add_time_sec=bool(args.add_time_sec),
                    compression=str(args.compression),
                    statistics=bool(args.statistics),
                    row_group_size=args.row_group_size,
                )
                _write_line(
                    "\t".join(
                        (
                            str(result.input_path),
                            str(result.output_path),
                            _format_bytes_gib(result.input_bytes),
                            _format_bytes_gib(result.output_bytes),
                            f"{result.read_wall_seconds:.3f}",
                            f"{result.read_cpu_seconds:.3f}",
                            f"{result.sink_wall_seconds:.3f}",
                            f"{result.sink_cpu_seconds:.3f}",
                            f"{result.approx_output_wall_seconds:.3f}",
                            f"{result.approx_output_cpu_seconds:.3f}",
                            f"{result.read_throughput_gib_per_s:.3f}",
                            f"{result.sink_throughput_gib_per_s:.3f}",
                            str(result.n_rows),
                        ),
                    ),
                )
            else:
                result = _bench_one(
                    input_path,
                    output_path,
                    symbols=args.symbols,
                    exchanges=args.exchanges,
                    use_market_hours=use_market_hours,
                    market_open_sec=int(args.market_open_sec),
                    market_close_sec=int(args.market_close_sec),
                    all_columns=bool(args.all_columns),
                    add_time_sec=bool(args.add_time_sec),
                    compression=str(args.compression),
                    statistics=bool(args.statistics),
                    row_group_size=args.row_group_size,
                )
                _write_line(
                    "\t".join(
                        (
                            str(result.input_path),
                            str(result.output_path),
                            _format_bytes_gib(result.input_bytes),
                            _format_bytes_gib(result.output_bytes),
                            f"{result.wall_seconds:.3f}",
                            f"{result.cpu_seconds:.3f}",
                            f"{result.throughput_gib_per_s:.3f}",
                        ),
                    ),
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
