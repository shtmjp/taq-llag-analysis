import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections.abc import Callable
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import ppllag

    return Callable, Path, np, pl, plt, ppllag


@app.cell
def _(Path, np):
    master_path = Path("data/dailyTAQ/MASTER/EQY_US_ALL_REF_MASTER_20251103")
    trade_base_dir = Path("data/filtered/trade")
    output_base_dir = Path("data/derived/cpcf_round_lot_10")
    target_dates = ["20251103", "20251030"]
    round_lot_target = 10
    traded_flag = 1
    obs_window = (35_100.0, 56_701.0)
    u_range = (-1.0, 1.0)
    bw_candidates = np.array([1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
    u_grid = np.linspace(-1e-3, 1e-3, 2001, dtype=np.float64)
    kernel = "tent"
    allow_not_simple = False
    return (
        allow_not_simple,
        bw_candidates,
        kernel,
        master_path,
        obs_window,
        output_base_dir,
        round_lot_target,
        target_dates,
        trade_base_dir,
        traded_flag,
        u_grid,
        u_range,
    )


@app.cell
def _(Callable, pl):
    TradeLazyPreprocessor = Callable[[pl.LazyFrame], pl.LazyFrame]

    def participant_timestamp_time_expr(
        timestamp_col: str = "Participant Timestamp",
    ) -> pl.Expr:
        timestamp = pl.col(timestamp_col)
        hh = (timestamp // 10_000_000_000_000).cast(pl.Float64)
        mm = ((timestamp // 100_000_000_000) % 100).cast(pl.Float64)
        ss = ((timestamp // 1_000_000_000) % 100).cast(pl.Float64)
        subsec = (timestamp % 1_000_000_000).cast(pl.Float64) / 1_000_000_000.0
        return (hh * 3600.0 + mm * 60.0 + ss + subsec).alias(
            "participant_timestamp_time",
        )

    def _all_trades(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    def _intermarket_sweep(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.filter(pl.col("Sale Condition").str.contains("F", literal=True))

    def collapse_same_timestamp_trades(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.group_by(["Exchange", "Participant Timestamp"]).agg(
            pl.col("Trade Volume").sum().alias("Trade Volume"),
            (
                (pl.col("Trade Price") * pl.col("Trade Volume")).sum()
                / pl.col("Trade Volume").sum()
            ).alias("Trade Price"),
        )

    trade_preprocessors: dict[str, TradeLazyPreprocessor] = {
        "all_trades": _all_trades,
        "intermarket_sweep": _intermarket_sweep,
    }
    preprocess_names = tuple(trade_preprocessors)
    preprocess_registry_df = pl.DataFrame({"preprocess_name": preprocess_names})
    return (
        collapse_same_timestamp_trades,
        participant_timestamp_time_expr,
        preprocess_names,
        preprocess_registry_df,
        trade_preprocessors,
    )


@app.cell
def _(master_path, pl, round_lot_target, traded_flag):
    master_candidates_df = (
        pl.read_csv(
            master_path,
            separator="|",
            has_header=True,
            comment_prefix="END",
            encoding="latin1",
        )
        .filter(
            (pl.col("Round_Lot") == round_lot_target)
            & (pl.col("TradedOnNasdaq") == traded_flag)
            & (pl.col("TradedOnBATS") == traded_flag),
        )
        .select(
            "Symbol",
            "Listed_Exchange",
            "Tape",
            "TradedOnNasdaq",
            "TradedOnBATS",
        )
        .sort("Symbol")
    )
    return (master_candidates_df,)


@app.cell
def _(master_path, pl, round_lot_target):
    round_lot_symbols_df = (
        pl.read_csv(
            master_path,
            separator="|",
            has_header=True,
            comment_prefix="END",
            encoding="latin1",
        )
        .filter(pl.col("Round_Lot") == round_lot_target)
        .select("Symbol", "Security_Description")
        .sort("Symbol")
    )
    round_lot_symbol_count = round_lot_symbols_df.height
    round_lot_symbol_list = round_lot_symbols_df.get_column("Symbol").to_list()
    return round_lot_symbol_count, round_lot_symbol_list, round_lot_symbols_df


@app.cell
def _(round_lot_symbol_count):
    round_lot_symbol_count  # noqa: B018
    return


@app.cell
def _(round_lot_symbols_df):
    round_lot_symbols_df  # noqa: B018
    return


@app.cell
def _(round_lot_symbol_list):
    round_lot_symbol_list  # noqa: B018
    return


@app.cell
def _(  # noqa: PLR0915
    allow_not_simple,
    bw_candidates,
    collapse_same_timestamp_trades,
    kernel,
    master_candidates_df,
    np,
    obs_window,
    output_base_dir,
    participant_timestamp_time_expr,
    pl,
    plt,
    ppllag,
    preprocess_names,
    target_dates,
    trade_base_dir,
    trade_preprocessors,
    u_grid,
    u_range,
):
    summary_rows: list[dict[str, object]] = []

    for candidate in master_candidates_df.iter_rows(named=True):
        symbol = candidate["Symbol"]
        for preprocess_name in preprocess_names:
            preprocess = trade_preprocessors[preprocess_name]
            for date_yyyymmdd in target_dates:
                parquet_path = trade_base_dir / symbol / f"trade_{date_yyyymmdd}.parquet"
                output_path = (
                    output_base_dir / preprocess_name / date_yyyymmdd / f"{symbol}_QZ_cpcf.png"
                )

                row: dict[str, object] = {
                    "preprocess_name": preprocess_name,
                    "date_yyyymmdd": date_yyyymmdd,
                    "Symbol": symbol,
                    "status": "",
                    "raw_n_q": 0,
                    "raw_n_z": 0,
                    "n_q": 0,
                    "n_z": 0,
                    "bandwidth": None,
                    "output_path": None,
                    "parquet_path": str(parquet_path),
                    "Listed_Exchange": candidate["Listed_Exchange"],
                    "Tape": candidate["Tape"],
                    "error": None,
                }

                if not parquet_path.exists():
                    row["status"] = "missing_parquet"
                    summary_rows.append(row)
                    continue

                preprocessed_lf = preprocess(pl.scan_parquet(parquet_path))
                raw_counts_df = (
                    preprocessed_lf.filter(pl.col("Exchange").is_in(["Q", "Z"]))
                    .group_by("Exchange")
                    .agg(pl.len().alias("n_trades"))
                    .collect(engine="streaming")
                )
                raw_count_by_exchange = {
                    exchange: int(n_trades) for exchange, n_trades in raw_counts_df.iter_rows()
                }
                row["raw_n_q"] = raw_count_by_exchange.get("Q", 0)
                row["raw_n_z"] = raw_count_by_exchange.get("Z", 0)

                trade_df = (
                    collapse_same_timestamp_trades(preprocessed_lf)
                    .with_columns(participant_timestamp_time_expr())
                    .select(["Exchange", "participant_timestamp_time"])
                    .sort(["Exchange", "participant_timestamp_time"])
                    .collect(engine="streaming")
                )

                data1 = np.sort(
                    np.asarray(
                        trade_df.filter(pl.col("Exchange") == "Q")
                        .get_column("participant_timestamp_time")
                        .to_numpy(),
                        dtype=np.float64,
                    ),
                )
                data2 = np.sort(
                    np.asarray(
                        trade_df.filter(pl.col("Exchange") == "Z")
                        .get_column("participant_timestamp_time")
                        .to_numpy(),
                        dtype=np.float64,
                    ),
                )

                row["n_q"] = int(data1.size)
                row["n_z"] = int(data2.size)

                if data1.size == 0 and data2.size == 0:
                    row["status"] = "missing_qz"
                    summary_rows.append(row)
                    continue
                if data1.size == 0:
                    row["status"] = "missing_q"
                    summary_rows.append(row)
                    continue
                if data2.size == 0:
                    row["status"] = "missing_z"
                    summary_rows.append(row)
                    continue

                try:
                    bandwidth = ppllag.lepski_bw_selector_for_cpcf_mode(
                        data1,
                        data2,
                        obs_window=obs_window,
                        u_range=u_range,
                        bw_candidates=bw_candidates.tolist(),
                        kernel=kernel,
                        allow_not_simple=allow_not_simple,
                    )
                    cpcf_values = ppllag.cpcf(
                        data1,
                        data2,
                        u_values=u_grid,
                        obs_window=obs_window,
                        bandwidth=bandwidth,
                        kernel=kernel,
                        allow_not_simple=allow_not_simple,
                    )
                    modes = ppllag.find_cpcf_modes(
                        data1,
                        data2,
                        u_range=u_range,
                        obs_window=obs_window,
                        bandwidth=bandwidth,
                        kernel=kernel,
                        allow_not_simple=allow_not_simple,
                    )

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.plot(u_grid, cpcf_values, color="#0f766e", linewidth=1.5)
                    ax.axvline(0.0, color="#57534e", linestyle="--", linewidth=1.0)
                    if modes.size > 0:
                        mode_values = np.interp(modes, u_grid, cpcf_values)
                        ax.scatter(modes, mode_values, color="#b91c1c", s=28, zorder=3)
                    ax.set_xlabel("Lag u (seconds)")
                    ax.set_ylabel("CPCF")
                    ax.set_title(
                        f"{symbol} CPCF (Q -> Z), {date_yyyymmdd}, {preprocess_name}, "
                        f"bandwidth={bandwidth:.4g}",
                    )
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    fig.savefig(output_path, dpi=200)
                    plt.close(fig)

                    row["status"] = "saved"
                    row["bandwidth"] = float(bandwidth)
                    row["output_path"] = str(output_path)
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = f"{type(exc).__name__}: {exc}"

                summary_rows.append(row)

    summary_df = pl.DataFrame(summary_rows).sort(
        ["preprocess_name", "date_yyyymmdd", "Symbol"],
    )
    aggregate_summary_df = (
        summary_df.group_by("preprocess_name", "date_yyyymmdd", "status")
        .agg(pl.len().alias("n_cases"))
        .sort(["preprocess_name", "date_yyyymmdd", "status"])
    )
    return aggregate_summary_df, summary_df


@app.cell
def _(preprocess_registry_df):
    preprocess_registry_df  # noqa: B018
    return


@app.cell
def _(master_candidates_df):
    master_candidates_df  # noqa: B018
    return


@app.cell
def _(summary_df):
    summary_df  # noqa: B018
    return


@app.cell
def _(aggregate_summary_df):
    aggregate_summary_df  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
