from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import polars as pl
import ppllag

from ._trade_quote_common import (
    ALLOW_NOT_SIMPLE,
    CROSS_K_SUMMARY_FILENAME,
    KERNEL,
    MODE_SUMMARY_OUTPUT_BASE_DIR,
    MODES_FILENAME,
    OBS_WINDOW,
    QUOTE_BASE_DIR,
    TRADE_BASE_DIR,
    _event_arrays_by_exchange,
)

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_SYMBOLS: tuple[str, ...] = ("MORN", "MAR", "MDB", "MSFT")
DEFAULT_DATES: tuple[str, ...] = ("20251031", "20251103")
DEFAULT_TRADE_EXCHANGE = "Q"
DEFAULT_QUOTE_EXCHANGE = "Z"
OUTPUT_BASE_DIR = Path("data/derived/trade_quote_cpcf_examples")
U_GRID_SPEC: dict[str, object] = {
    "start": -1e-3,
    "stop": 1e-3,
    "num": 2001,
    "dtype": "float64",
}
FULL_MODE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "bandwidth",
    "closest_mode_to_zero_sec",
)


def _u_grid() -> np.ndarray:
    return np.linspace(
        float(U_GRID_SPEC["start"]),
        float(U_GRID_SPEC["stop"]),
        int(U_GRID_SPEC["num"]),
        dtype=np.float64,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_output(args: Sequence[str]) -> str | None:
    completed = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_metadata() -> dict[str, object]:
    git_sha = _git_output(("git", "rev-parse", "HEAD"))
    git_status = _git_output(("git", "status", "--short"))
    return {
        "commit_sha": git_sha,
        "dirty": bool(git_status),
    }


def _cross_k_summary_frame(run_dir: Path) -> pl.DataFrame:
    return pl.read_csv(
        run_dir / CROSS_K_SUMMARY_FILENAME,
        schema_overrides={"date_yyyymmdd": pl.String},
    )


def _modes_frame(run_dir: Path) -> pl.DataFrame:
    return pl.read_csv(
        run_dir / MODES_FILENAME,
        schema_overrides={"date_yyyymmdd": pl.String},
    )


def _validate_cpcf_source_run(
    run_dir: Path,
    cross_k_summary_df: pl.DataFrame,
) -> None:
    missing_columns = sorted(
        column_name
        for column_name in FULL_MODE_REQUIRED_COLUMNS
        if column_name not in cross_k_summary_df.columns
    )
    modes_path = run_dir / MODES_FILENAME

    if not modes_path.exists() and missing_columns:
        missing_text = ", ".join(missing_columns)
        msg = (
            "CPCF example plots require a full mode-summary run, but "
            f"{run_dir} looks like a cross-K-only output: {MODES_FILENAME} is missing "
            f"and {CROSS_K_SUMMARY_FILENAME} does not contain {missing_text}."
        )
        raise ValueError(msg)
    if not modes_path.exists():
        msg = f"CPCF example plots require {MODES_FILENAME} in {run_dir}, but it was not found."
        raise ValueError(msg)
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        msg = (
            "CPCF example plots require a full mode-summary run, but "
            f"{CROSS_K_SUMMARY_FILENAME} in {run_dir} is missing {missing_text}."
        )
        raise ValueError(msg)


def _cross_k_row(
    cross_k_summary_df: pl.DataFrame,
    *,
    symbol: str,
    date_yyyymmdd: str,
    trade_exchange: str,
    quote_exchange: str,
) -> dict[str, object]:
    match_df = cross_k_summary_df.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("date_yyyymmdd") == date_yyyymmdd)
        & (pl.col("trade_exchange") == trade_exchange)
        & (pl.col("quote_exchange") == quote_exchange)
    )
    if match_df.height != 1:
        msg = (
            "Expected one cross-k summary row for "
            f"{symbol=} {date_yyyymmdd=} {trade_exchange=} {quote_exchange=}, "
            f"found {match_df.height}."
        )
        raise ValueError(msg)
    row = match_df.row(0, named=True)
    if row["status"] != "ok":
        msg = (
            "CPCF example plots require status='ok' rows, but found "
            f"{row['status']!r} for {symbol=} {date_yyyymmdd=} "
            f"{trade_exchange=} {quote_exchange=}."
        )
        raise ValueError(msg)
    return row


def _mode_candidates_sec(
    modes_df: pl.DataFrame,
    *,
    symbol: str,
    date_yyyymmdd: str,
    trade_exchange: str,
    quote_exchange: str,
) -> list[float]:
    mode_values = (
        modes_df.filter(
            (pl.col("symbol") == symbol)
            & (pl.col("date_yyyymmdd") == date_yyyymmdd)
            & (pl.col("trade_exchange") == trade_exchange)
            & (pl.col("quote_exchange") == quote_exchange)
        )
        .get_column("mode_sec")
        .to_list()
    )
    return sorted({float(mode_sec) for mode_sec in mode_values})


def _event_arrays(
    *,
    symbol: str,
    date_yyyymmdd: str,
    trade_exchange: str,
    quote_exchange: str,
) -> tuple[np.ndarray, np.ndarray]:
    trade_path = TRADE_BASE_DIR / symbol / f"trade_{date_yyyymmdd}.parquet"
    quote_path = QUOTE_BASE_DIR / symbol / f"quote_{date_yyyymmdd}.parquet"
    trade_arrays = _event_arrays_by_exchange(
        trade_path,
        "Participant Timestamp",
        (trade_exchange,),
        obs_window=OBS_WINDOW,
    )
    quote_arrays = _event_arrays_by_exchange(
        quote_path,
        "Participant_Timestamp",
        (quote_exchange,),
        obs_window=OBS_WINDOW,
    )
    return trade_arrays[trade_exchange], quote_arrays[quote_exchange]


def _date_label(date_yyyymmdd: str) -> str:
    return f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"


def _microsecond_label(value_sec: float) -> str:
    return f"{value_sec * 1_000_000.0:.1f} us"


def _plot_symbol(
    *,
    symbol: str,
    date_panels: Sequence[dict[str, object]],
    trade_exchange: str,
    quote_exchange: str,
    outpath: Path,
) -> Path:
    figure, axes = plt.subplots(
        1,
        len(date_panels),
        figsize=(12, 4.6),
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes)

    y_min = min(float(np.min(panel["cpcf_values"])) for panel in date_panels)
    y_max = max(float(np.max(panel["cpcf_values"])) for panel in date_panels)
    y_pad = max(0.02 * (y_max - y_min), 1e-6)

    for index, (axis, panel) in enumerate(zip(axes_array, date_panels, strict=True)):
        u_values_us = np.asarray(panel["u_values"], dtype=np.float64) * 1_000_000.0
        cpcf_values = np.asarray(panel["cpcf_values"], dtype=np.float64)
        mode_candidates_sec = list(panel["mode_candidates_sec"])
        closest_mode_sec = float(panel["closest_mode_to_zero_sec"])

        axis.plot(u_values_us, cpcf_values, color="#1d4ed8", linewidth=2.0)
        axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
        for mode_sec in mode_candidates_sec:
            axis.axvline(
                mode_sec * 1_000_000.0,
                color="#94a3b8",
                linewidth=1.0,
                alpha=0.9,
                zorder=0,
            )
        axis.axvline(
            closest_mode_sec * 1_000_000.0,
            color="#dc2626",
            linewidth=2.2,
            label="closest mode" if index == 0 else None,
        )
        axis.set_xlim(float(u_values_us[0]), float(u_values_us[-1]))
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.ticklabel_format(style="plain", axis="x")
        axis.grid(alpha=0.2)
        axis.set_xlabel("u (microseconds)")
        axis.set_title(
            "\n".join(
                (
                    _date_label(str(panel["date_yyyymmdd"])),
                    f"bw={float(panel['bandwidth']):.1e}",
                    f"closest={_microsecond_label(closest_mode_sec)}",
                )
            ),
            fontsize=10,
        )

    axes_array[0].set_ylabel("CPCF")
    axes_array[0].legend(loc="upper right", fontsize=8, frameon=False)
    figure.suptitle(
        f"{symbol} (trade={trade_exchange}, quote={quote_exchange})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(outpath, dpi=200)
    plt.close(figure)
    return outpath


def build_cpcf_examples(
    *,
    run_id: str,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    dates: Sequence[str] = DEFAULT_DATES,
    trade_exchange: str = DEFAULT_TRADE_EXCHANGE,
    quote_exchange: str = DEFAULT_QUOTE_EXCHANGE,
    output_base_dir: Path = OUTPUT_BASE_DIR,
) -> Path:
    """Write CPCF example plots for selected symbols and dates.

    Parameters
    ----------
    run_id
        Existing mode-summary run identifier used to load the chosen bandwidths
        and mode candidates.
    symbols
        Symbols to plot. Each symbol is saved as one PNG with one panel per date.
    dates
        Trading dates formatted as ``YYYYMMDD``.
    trade_exchange
        Exchange code used on the trade side.
    quote_exchange
        Exchange code used on the quote side.
    output_base_dir
        Parent directory for the CPCF example output directory.

    Returns
    -------
    pathlib.Path
        Output directory containing the PNG figures and manifest files.

    """
    source_run_dir = MODE_SUMMARY_OUTPUT_BASE_DIR / run_id
    output_dir = output_base_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cross_k_summary_df = _cross_k_summary_frame(source_run_dir)
    _validate_cpcf_source_run(source_run_dir, cross_k_summary_df)
    modes_df = _modes_frame(source_run_dir)
    u_values = _u_grid()

    sample_rows: list[dict[str, object]] = []
    png_paths: list[str] = []

    for symbol in symbols:
        date_panels: list[dict[str, object]] = []
        png_path = output_dir / f"cpcf__{symbol}.png"
        for date_yyyymmdd in dates:
            cross_k_row = _cross_k_row(
                cross_k_summary_df,
                symbol=symbol,
                date_yyyymmdd=date_yyyymmdd,
                trade_exchange=trade_exchange,
                quote_exchange=quote_exchange,
            )
            mode_candidates_sec = _mode_candidates_sec(
                modes_df,
                symbol=symbol,
                date_yyyymmdd=date_yyyymmdd,
                trade_exchange=trade_exchange,
                quote_exchange=quote_exchange,
            )
            trade_event_times, quote_event_times = _event_arrays(
                symbol=symbol,
                date_yyyymmdd=date_yyyymmdd,
                trade_exchange=trade_exchange,
                quote_exchange=quote_exchange,
            )
            bandwidth = float(cross_k_row["bandwidth"])
            closest_mode_to_zero_sec = float(cross_k_row["closest_mode_to_zero_sec"])
            cpcf_values = np.asarray(
                ppllag.cpcf(
                    trade_event_times,
                    quote_event_times,
                    u_values,
                    OBS_WINDOW,
                    bandwidth=bandwidth,
                    kernel=KERNEL,
                    allow_not_simple=ALLOW_NOT_SIMPLE,
                ),
                dtype=np.float64,
            )

            date_panels.append(
                {
                    "date_yyyymmdd": date_yyyymmdd,
                    "u_values": u_values,
                    "cpcf_values": cpcf_values,
                    "bandwidth": bandwidth,
                    "closest_mode_to_zero_sec": closest_mode_to_zero_sec,
                    "mode_candidates_sec": mode_candidates_sec,
                }
            )
            sample_rows.append(
                {
                    "run_id": run_id,
                    "date_yyyymmdd": date_yyyymmdd,
                    "symbol": symbol,
                    "trade_exchange": trade_exchange,
                    "quote_exchange": quote_exchange,
                    "bandwidth": bandwidth,
                    "closest_mode_to_zero_sec": closest_mode_to_zero_sec,
                    "n_trade_events": int(trade_event_times.size),
                    "n_quote_events": int(quote_event_times.size),
                    "mode_count": len(mode_candidates_sec),
                    "mode_candidates_sec_json": json.dumps(mode_candidates_sec),
                    "png_path": str(png_path),
                }
            )

        _plot_symbol(
            symbol=symbol,
            date_panels=date_panels,
            trade_exchange=trade_exchange,
            quote_exchange=quote_exchange,
            outpath=png_path,
        )
        png_paths.append(str(png_path))

    sample_manifest_df = pl.from_dicts(sample_rows).sort(["symbol", "date_yyyymmdd"])
    sample_manifest_path = output_dir / "sample_manifest.csv"
    sample_manifest_df.write_csv(sample_manifest_path)

    run_manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run_dir": str(source_run_dir),
        "cross_k_summary_csv": str(source_run_dir / CROSS_K_SUMMARY_FILENAME),
        "modes_csv": str(source_run_dir / MODES_FILENAME),
        "output_dir": str(output_dir),
        "trade_base_dir": str(TRADE_BASE_DIR),
        "quote_base_dir": str(QUOTE_BASE_DIR),
        "symbols": list(symbols),
        "dates": list(dates),
        "trade_exchange": trade_exchange,
        "quote_exchange": quote_exchange,
        "obs_window": list(OBS_WINDOW),
        "kernel": KERNEL,
        "allow_not_simple": ALLOW_NOT_SIMPLE,
        "u_grid_spec": U_GRID_SPEC,
        "package_versions": {
            "matplotlib": _package_version("matplotlib"),
            "numpy": _package_version("numpy"),
            "polars": _package_version("polars"),
            "ppllag": _package_version("ppllag"),
        },
        "git": _git_metadata(),
        "sample_manifest_csv": str(sample_manifest_path),
        "png_paths": png_paths,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    """Run CPCF example generation from the command line.

    Returns
    -------
    int
        Process exit code.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        required=True,
        help="Existing mode-summary run identifier.",
    )
    parser.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        help="Exact symbol to plot. Repeat to pass multiple symbols.",
    )
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        help="Trading date formatted as YYYYMMDD. Repeat to pass multiple dates.",
    )
    parser.add_argument(
        "--trade-exchange",
        default=DEFAULT_TRADE_EXCHANGE,
        help="Exchange code for the trade side.",
    )
    parser.add_argument(
        "--quote-exchange",
        default=DEFAULT_QUOTE_EXCHANGE,
        help="Exchange code for the quote side.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=OUTPUT_BASE_DIR,
        help="Parent directory for the output run directory.",
    )
    args = parser.parse_args()

    output_dir = build_cpcf_examples(
        run_id=args.run_id,
        symbols=tuple(args.symbols) if args.symbols else DEFAULT_SYMBOLS,
        dates=tuple(args.dates) if args.dates else DEFAULT_DATES,
        trade_exchange=args.trade_exchange,
        quote_exchange=args.quote_exchange,
        output_base_dir=args.output_base_dir,
    )
    summary = {
        "output_dir": str(output_dir),
        "run_manifest_json": str(output_dir / "run_manifest.json"),
        "sample_manifest_csv": str(output_dir / "sample_manifest.csv"),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
