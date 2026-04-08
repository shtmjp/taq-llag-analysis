import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    import pandas as pd

    return Path, json, pl, plt


@app.cell
def _(Path):
    results_base_dir = Path("data/derived/m_trade_quote_modes")
    master_base_dir = Path("data/dailyTAQ/MASTER")
    run_id = ""
    trade_exchange = "Q"
    quote_exchange = "Z"
    target_dates = (20251031, 20251103)
    mode_column = "closest_mode_to_zero_sec"
    return (
        master_base_dir,
        mode_column,
        quote_exchange,
        results_base_dir,
        target_dates,
        trade_exchange,
    )


@app.cell
def _(json, master_base_dir, pl, results_base_dir, target_dates):
    available_run_ids = (
        sorted(path.name for path in results_base_dir.iterdir() if path.is_dir())
        if results_base_dir.exists()
        else []
    )
    resolved_run_id = available_run_ids[-1]
    run_dir = results_base_dir / resolved_run_id
    run_files = {
        "run_config_json": run_dir / "run_config.json",
        "pair_summary_csv": run_dir / "pair_summary.csv",
    }
    master_files = {
        date_yyyymmdd: master_base_dir / f"EQY_US_ALL_REF_MASTER_{date_yyyymmdd}"
        for date_yyyymmdd in target_dates
    }

    run_config = json.loads(run_files["run_config_json"].read_text(encoding="utf-8"))
    pair_summary_df = pl.read_csv(run_files["pair_summary_csv"])
    run_config
    return master_files, pair_summary_df


@app.cell
def _(pair_summary_df, pl, quote_exchange, target_dates, trade_exchange):
    selected_pair_df = pair_summary_df.filter(
        (pl.col("status") == "ok")
        & (pl.col("trade_exchange") == trade_exchange)
        & (pl.col("quote_exchange") == quote_exchange)
        & (pl.col("date_yyyymmdd").is_in(target_dates))
    ).sort(["date_yyyymmdd", "symbol"])
    selected_pair_df
    return (selected_pair_df,)


@app.cell
def _(master_files, pl):
    _master_frames = []
    for _date_yyyymmdd, _path in master_files.items():
        _master_frames.append(
            pl.read_csv(
                _path,
                separator="|",
                has_header=True,
                comment_prefix="END",
                encoding="latin1",
            )
            .select(
                pl.col("Symbol").alias("symbol"),
                pl.col("Round_Lot").alias("round_lot"),
            )
            .with_columns(pl.lit(int(_date_yyyymmdd)).alias("date_yyyymmdd"))
        )

    master_round_lot_df = pl.concat(_master_frames, how="vertical").sort(
        ["date_yyyymmdd", "symbol"],
    )
    return (master_round_lot_df,)


@app.cell
def _(master_round_lot_df, selected_pair_df):
    selected_pair_with_round_lot_df = selected_pair_df.join(
        master_round_lot_df,
        on=["date_yyyymmdd", "symbol"],
        how="left",
    ).sort(["date_yyyymmdd", "round_lot", "symbol"])
    selected_pair_with_round_lot_df
    return (selected_pair_with_round_lot_df,)


@app.cell
def _(pl, quote_exchange, selected_pair_with_round_lot_df, trade_exchange):
    selected_pair_counts_df = (
        selected_pair_with_round_lot_df.group_by("date_yyyymmdd")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("symbol").n_unique().alias("n_symbols"),
            pl.col("round_lot").n_unique().alias("n_round_lot_buckets"),
        )
        .with_columns(
            pl.lit(trade_exchange).alias("trade_exchange"),
            pl.lit(quote_exchange).alias("quote_exchange"),
        )
        .sort("date_yyyymmdd")
    )
    selected_pair_counts_df
    return


@app.cell
def _(pl, selected_pair_with_round_lot_df):
    per_date_round_lot_counts_df = (
        selected_pair_with_round_lot_df.group_by(["date_yyyymmdd", "round_lot"])
        .agg(pl.len().alias("n_symbols"))
        .sort(["date_yyyymmdd", "round_lot"])
    )
    per_date_round_lot_counts_df
    return


@app.cell
def _(mode_column, pl, selected_pair_with_round_lot_df):
    per_date_round_lot_summary_df = (
        selected_pair_with_round_lot_df.group_by(["date_yyyymmdd", "round_lot"])
        .agg(
            pl.len().alias("n_symbols"),
            pl.col(mode_column).median().alias("mode_median_sec"),
            pl.col(mode_column).mean().alias("mode_mean_sec"),
            pl.col(mode_column).std().alias("mode_std_sec"),
        )
        .sort(["date_yyyymmdd", "round_lot"])
    )
    per_date_round_lot_summary_df
    return


@app.cell
def _(mode_column, pl, plt, selected_pair_with_round_lot_df, target_dates):
    _fig, _axes = plt.subplots(
        1,
        len(target_dates),
        figsize=(6 * len(target_dates), 4.5),
        sharey=True,
    )
    if len(target_dates) == 1:
        _axes = [_axes]

    for _axis, _date_yyyymmdd in zip(_axes, target_dates, strict=True):
        _date_df = selected_pair_with_round_lot_df.filter(
            pl.col("date_yyyymmdd") == _date_yyyymmdd,
        )
        _round_lots = sorted(_date_df.get_column("round_lot").unique().to_list())
        _values = [
            _date_df.filter(pl.col("round_lot") == _round_lot).get_column(mode_column).to_list()
            for _round_lot in _round_lots
        ]
        _counts = [len(_series) for _series in _values]

        if _values:
            _boxplot = _axis.boxplot(
                _values,
                tick_labels=[str(_round_lot) for _round_lot in _round_lots],
                patch_artist=True,
            )
            for _patch in _boxplot["boxes"]:
                _patch.set_facecolor("#bfdbfe")
                _patch.set_alpha(0.9)

            _y_min, _y_max = _axis.get_ylim()
            _y_span = _y_max - _y_min
            _y_text = _y_max - 0.06 * (_y_span if _y_span > 0.0 else 1.0)
            for _index, _count in enumerate(_counts, start=1):
                _axis.text(
                    _index,
                    _y_text,
                    f"n={_count}",
                    ha="center",
                    va="top",
                    fontsize=9,
                )
        else:
            _axis.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=_axis.transAxes,
            )

        _axis.set_title(str(_date_yyyymmdd))
        _axis.set_xlabel("Round lot")
        _axis.grid(alpha=0.2)

    _axes[0].set_ylabel("Closest mode to zero (seconds)")
    _fig.suptitle("Per-date mode distribution by round lot")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
