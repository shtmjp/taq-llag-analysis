import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from taq_llag_analysis.preprocess.daily_taq_paths import master_input_path

    plt.rcParams["axes.unicode_minus"] = False
    return Path, master_input_path, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    ## Overview

    This notebook inspects the distribution of `mode_sec` from
    `data/derived/trade_quote_modes/my-run/modes.csv`.

    It provides lightweight filters by date and exchange, descriptive
    statistics for the current selection, and distribution plots for the
    signed mode location and the magnitude of `|mode_sec|`.
    """)
    return


@app.cell
def _(Path):
    repo_root = Path(__file__).resolve().parents[1]
    input_csv_path = repo_root / "data" / "derived" / "trade_quote_modes" / "my-run" / "modes.csv"
    return (input_csv_path,)


@app.cell
def _(input_csv_path, master_input_path, pd):
    modes_df = pd.read_csv(
        input_csv_path,
        dtype={
            "date_yyyymmdd": "string",
            "symbol": "string",
            "trade_exchange": "string",
            "quote_exchange": "string",
        },
    )
    listed_exchange_frames = []
    for date_yyyymmdd in sorted(modes_df["date_yyyymmdd"].dropna().unique().tolist()):
        master_df = pd.read_csv(
            master_input_path(date_yyyymmdd),
            sep="|",
            encoding="latin1",
            compression="gzip",
            usecols=["Symbol", "Listed_Exchange"],
        )
        master_df = master_df.loc[master_df["Symbol"] != "END"].copy()
        master_df = master_df.rename(
            columns={
                "Symbol": "symbol",
                "Listed_Exchange": "listed_exchange",
            }
        )
        master_df["date_yyyymmdd"] = date_yyyymmdd
        listed_exchange_frames.append(master_df)

    listed_exchange_df = pd.concat(listed_exchange_frames, ignore_index=True)
    modes_df = modes_df.merge(
        listed_exchange_df,
        on=["date_yyyymmdd", "symbol"],
        how="left",
        validate="m:1",
    )
    listed_is_trade_exchange = modes_df["listed_exchange"].eq(modes_df["trade_exchange"])
    listed_is_trade_exchange = listed_is_trade_exchange.where(
        modes_df["listed_exchange"].notna(),
        pd.NA,
    ).astype("boolean")
    modes_df = modes_df.assign(
        exchange_pair=(
            modes_df["trade_exchange"].astype(str) + "-" + modes_df["quote_exchange"].astype(str)
        ),
        listed_is_trade_exchange=listed_is_trade_exchange,
        mode_ms=modes_df["mode_sec"] * 1000.0,
        mode_abs_sec=modes_df["mode_sec"].abs(),
    )
    return (modes_df,)


@app.cell
def _(mo, modes_df):
    date_options = ["All", *sorted(modes_df["date_yyyymmdd"].dropna().unique().tolist())]
    trade_exchange_options = [
        "All",
        *sorted(modes_df["trade_exchange"].dropna().unique().tolist()),
    ]
    quote_exchange_options = [
        "All",
        *sorted(modes_df["quote_exchange"].dropna().unique().tolist()),
    ]
    listed_trade_side_options = ["All", "Yes", "No"]

    date_selector = mo.ui.dropdown(
        options=date_options,
        value="All",
        label="Date",
    )
    trade_exchange_selector = mo.ui.dropdown(
        options=trade_exchange_options,
        value="All",
        label="Trade exchange",
    )
    quote_exchange_selector = mo.ui.dropdown(
        options=quote_exchange_options,
        value="All",
        label="Quote exchange",
    )
    listed_trade_side_selector = mo.ui.dropdown(
        options=listed_trade_side_options,
        value="All",
        label="Listed exchange on trade side",
    )
    signed_hist_limit_ms = mo.ui.slider(
        start=0.1,
        stop=1000.0,
        step=0.1,
        value=10.0,
        show_value=True,
        include_input=True,
        label="Signed histogram half-range (ms)",
    )
    top_n_mode_indices = mo.ui.slider(
        start=10,
        stop=50,
        step=5,
        value=20,
        show_value=True,
        label="Top mode indices shown",
    )
    top_n_exchange_pairs = mo.ui.slider(
        start=10,
        stop=50,
        step=5,
        value=20,
        show_value=True,
        label="Top exchange pairs shown",
    )
    mo.vstack(
        [
            mo.md("## Filters"),
            mo.hstack(
                [
                    date_selector,
                    trade_exchange_selector,
                    quote_exchange_selector,
                    listed_trade_side_selector,
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    signed_hist_limit_ms,
                    top_n_mode_indices,
                    top_n_exchange_pairs,
                ],
                justify="start",
            ),
        ]
    )
    return (
        date_selector,
        listed_trade_side_selector,
        quote_exchange_selector,
        signed_hist_limit_ms,
        top_n_exchange_pairs,
        top_n_mode_indices,
        trade_exchange_selector,
    )


@app.cell
def _(
    date_selector,
    listed_trade_side_selector,
    modes_df,
    quote_exchange_selector,
    trade_exchange_selector,
):
    filtered_modes_df = modes_df

    if date_selector.value != "All":
        filtered_modes_df = filtered_modes_df.loc[
            filtered_modes_df["date_yyyymmdd"] == date_selector.value
        ]
    if trade_exchange_selector.value != "All":
        filtered_modes_df = filtered_modes_df.loc[
            filtered_modes_df["trade_exchange"] == trade_exchange_selector.value
        ]
    if quote_exchange_selector.value != "All":
        filtered_modes_df = filtered_modes_df.loc[
            filtered_modes_df["quote_exchange"] == quote_exchange_selector.value
        ]
    listed_trade_side_mask = filtered_modes_df["listed_is_trade_exchange"]
    if listed_trade_side_selector.value == "Yes":
        filtered_modes_df = filtered_modes_df.loc[
            listed_trade_side_mask.notna() & listed_trade_side_mask
        ]
    if listed_trade_side_selector.value == "No":
        filtered_modes_df = filtered_modes_df.loc[
            listed_trade_side_mask.notna() & (~listed_trade_side_mask)
        ]

    filtered_modes_df = filtered_modes_df.reset_index(drop=True)
    selection_label = (
        f"date={date_selector.value}, "
        f"trade_exchange={trade_exchange_selector.value}, "
        f"quote_exchange={quote_exchange_selector.value}, "
        f"listed_exchange_on_trade_side={listed_trade_side_selector.value}"
    )
    return filtered_modes_df, selection_label


@app.cell
def _(filtered_modes_df, mo, selection_label):
    mo.md(
        f"""
        ## Current selection

        `{selection_label}`

        Rows: `{len(filtered_modes_df):,}`
        """
    )
    return


@app.cell
def _(filtered_modes_df, mo, pd):
    selection_summary_df = pd.DataFrame(
        {
            "metric": [
                "rows",
                "unique_symbols",
                "unique_dates",
                "unique_exchange_pairs",
                "min_mode_index",
                "max_mode_index",
                "mode_sec_mean",
                "mode_sec_std",
            ],
            "value": [
                len(filtered_modes_df),
                filtered_modes_df["symbol"].nunique(),
                filtered_modes_df["date_yyyymmdd"].nunique(),
                filtered_modes_df["exchange_pair"].nunique(),
                filtered_modes_df["mode_index"].min(),
                filtered_modes_df["mode_index"].max(),
                filtered_modes_df["mode_sec"].mean(),
                filtered_modes_df["mode_sec"].std(),
            ],
        }
    )
    mo.ui.table(selection_summary_df)
    return


@app.cell
def _(filtered_modes_df, mo, pd):
    listed_trade_side_summary_df = (
        filtered_modes_df["listed_is_trade_exchange"]
        .astype("string")
        .fillna("Missing")
        .replace(
            {
                "True": "Yes",
                "False": "No",
            }
        )
        .rename("listed_exchange_on_trade_side")
        .to_frame()
        .groupby("listed_exchange_on_trade_side", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(
            by="listed_exchange_on_trade_side",
            key=lambda series: series.map({"Yes": 0, "No": 1, "Missing": 2}),
            ignore_index=True,
        )
    )
    share_denominator = len(filtered_modes_df)
    if share_denominator > 0:
        listed_trade_side_summary_df["share"] = (
            listed_trade_side_summary_df["count"] / share_denominator
        )
    else:
        listed_trade_side_summary_df["share"] = pd.Series(dtype="float64")
    mo.ui.table(listed_trade_side_summary_df, label="Listed exchange on trade side summary")
    return


@app.cell
def _(filtered_modes_df, mo):
    quantile_levels = [0.0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    mode_sec_quantiles_df = (
        filtered_modes_df["mode_sec"]
        .quantile(quantile_levels)
        .rename_axis("quantile")
        .reset_index(name="mode_sec")
    )
    mo.ui.table(mode_sec_quantiles_df)
    return


@app.cell
def _(filtered_modes_df, mo, np, top_n_mode_indices):
    mode_index_counts_df = (
        filtered_modes_df.groupby("mode_index", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("mode_index", ignore_index=True)
    )
    if len(filtered_modes_df) > 0:
        mode_index_counts_df["share"] = mode_index_counts_df["count"] / len(filtered_modes_df)
    else:
        mode_index_counts_df["share"] = np.nan
    mode_index_counts_head_df = mode_index_counts_df.head(int(top_n_mode_indices.value))
    mo.ui.table(mode_index_counts_head_df)
    return (mode_index_counts_head_df,)


@app.cell
def _(filtered_modes_df, mo, np, top_n_exchange_pairs):
    exchange_pair_counts_df = (
        filtered_modes_df.groupby(
            ["exchange_pair", "trade_exchange", "quote_exchange"], as_index=False
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(
            by=["count", "exchange_pair"],
            ascending=[False, True],
            ignore_index=True,
        )
    )
    if len(filtered_modes_df) > 0:
        exchange_pair_counts_df["share"] = exchange_pair_counts_df["count"] / len(filtered_modes_df)
    else:
        exchange_pair_counts_df["share"] = np.nan
    exchange_pair_counts_head_df = exchange_pair_counts_df.head(int(top_n_exchange_pairs.value))
    mo.ui.table(exchange_pair_counts_head_df)
    return (exchange_pair_counts_head_df,)


@app.cell
def _(filtered_modes_df, mo):
    date_summary_df = (
        filtered_modes_df.groupby("date_yyyymmdd", as_index=False)
        .agg(
            rows=("mode_sec", "size"),
            unique_symbols=("symbol", "nunique"),
            unique_exchange_pairs=("exchange_pair", "nunique"),
            mode_sec_median=("mode_sec", "median"),
            mode_sec_p01=("mode_sec", lambda series: series.quantile(0.01)),
            mode_sec_p99=("mode_sec", lambda series: series.quantile(0.99)),
        )
        .sort_values("date_yyyymmdd", ignore_index=True)
    )
    mo.ui.table(date_summary_df)
    return


@app.cell
def _(
    exchange_pair_counts_head_df,
    filtered_modes_df,
    mode_index_counts_head_df,
    np,
    plt,
    signed_hist_limit_ms,
):
    signed_hist_limit_sec = signed_hist_limit_ms.value / 1000.0
    signed_hist_window_df = filtered_modes_df.loc[
        filtered_modes_df["mode_sec"].abs() <= signed_hist_limit_sec
    ]
    log_abs_mode_sec = np.log10(
        filtered_modes_df.loc[filtered_modes_df["mode_sec"] != 0.0, "mode_abs_sec"].to_numpy()
    )
    outside_signed_hist_share = np.nan
    if len(filtered_modes_df) > 0:
        outside_signed_hist_share = 1.0 - (len(signed_hist_window_df) / len(filtered_modes_df))

    _, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    axes[0, 0].hist(
        signed_hist_window_df["mode_ms"],
        bins=5000,
        range=(-signed_hist_limit_ms.value, signed_hist_limit_ms.value),
        color="#33658A",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0, 0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title(
        "Signed mode histogram "
        f"within +/- {signed_hist_limit_ms.value:.1f} ms\n"
        f"outside-window share: {outside_signed_hist_share:.2%}"
        if len(filtered_modes_df) > 0
        else "Signed mode histogram (empty selection)"
    )
    axes[0, 0].set_xlabel("mode_sec (ms)")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_xlim(-0.1, 0.1)

    axes[0, 1].hist(
        log_abs_mode_sec,
        bins=120,
        color="#55A630",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0, 1].set_title("Magnitude histogram of log10(|mode_sec|)")
    axes[0, 1].set_xlabel("log10(|mode_sec|) [sec]")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].bar(
        mode_index_counts_head_df["mode_index"].astype(str),
        mode_index_counts_head_df["count"],
        color="#BC4749",
    )
    axes[1, 0].set_title("Mode-index counts (lowest indices)")
    axes[1, 0].set_xlabel("mode_index")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].tick_params(axis="x", labelrotation=90)

    axes[1, 1].barh(
        exchange_pair_counts_head_df["exchange_pair"][::-1],
        exchange_pair_counts_head_df["count"][::-1],
        color="#F4A259",
    )
    axes[1, 1].set_title("Top exchange pairs by row count")
    axes[1, 1].set_xlabel("count")
    axes[1, 1].set_ylabel("exchange_pair")

    plt.gcf()
    return


@app.cell
def _(filtered_modes_df, mo):
    sample_columns = [
        "date_yyyymmdd",
        "symbol",
        "listed_exchange",
        "listed_is_trade_exchange",
        "trade_exchange",
        "quote_exchange",
        "mode_index",
        "mode_sec",
    ]
    mo.ui.table(filtered_modes_df.loc[:, sample_columns].head(50), label="Sample rows")
    return


if __name__ == "__main__":
    app.run()
