import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import glob
    from pathlib import Path

    import pandas as pd

    return Path, glob, pd


@app.cell
def _(Path):
    results_base_dir = Path("data/derived/m_trade_quote_modes")
    master_base_dir = Path("data/dailyTAQ/MASTER")
    cta_glob = "data/cta_symbol_files_202509/CTA.Symbol.File.202509*.csv"
    run_id = ""
    trade_exchange = "Q"
    quote_exchange = "Z"
    target_dates = (20251031, 20251103)
    mode_column = "closest_mode_to_zero_sec"
    cross_k_columns = [
        "cross_k_neg_1e3_1e4",
        "cross_k_neg_1e4_1e5",
        "cross_k_neg_1e5_0",
        "cross_k_pos_0_1e5",
        "cross_k_pos_1e5_1e4",
        "cross_k_pos_1e4_1e3",
    ]
    return (
        cross_k_columns,
        cta_glob,
        master_base_dir,
        mode_column,
        quote_exchange,
        results_base_dir,
        run_id,
        target_dates,
        trade_exchange,
    )


@app.cell
def _(master_base_dir, results_base_dir, run_id, target_dates):
    _available_run_ids = (
        sorted(path.name for path in results_base_dir.iterdir() if path.is_dir())
        if results_base_dir.exists()
        else []
    )
    if run_id:
        resolved_run_id = run_id
    elif _available_run_ids:
        resolved_run_id = _available_run_ids[-1]
    else:
        raise FileNotFoundError(
            f"No run directories found under {results_base_dir}.",
        )

    run_dir = results_base_dir / resolved_run_id
    pair_summary_path = run_dir / "pair_summary.csv"
    master_paths = {
        date_yyyymmdd: master_base_dir / f"EQY_US_ALL_REF_MASTER_{date_yyyymmdd}"
        for date_yyyymmdd in target_dates
    }
    left_date, right_date = target_dates
    return left_date, master_paths, pair_summary_path, right_date


@app.cell
def _(
    cross_k_columns,
    left_date,
    master_paths,
    mode_column,
    pair_summary_path,
    pd,
    quote_exchange,
    right_date,
    trade_exchange,
):
    pair_summary_df = pd.read_csv(pair_summary_path)
    _selected_pair_df = pair_summary_df.loc[
        (pair_summary_df["status"] == "ok")
        & (pair_summary_df["trade_exchange"] == trade_exchange)
        & (pair_summary_df["quote_exchange"] == quote_exchange)
        & (pair_summary_df["date_yyyymmdd"].isin([left_date, right_date])),
        [
            "symbol",
            "date_yyyymmdd",
            mode_column,
            "trade_exchange",
            "quote_exchange",
            *cross_k_columns,
        ],
    ].copy()

    _master_frames = []
    for _date_yyyymmdd, _path in master_paths.items():
        _master_df = pd.read_csv(
            _path,
            sep="|",
            encoding="latin1",
            usecols=["Symbol", "Round_Lot"],
        )
        _master_df = _master_df.loc[_master_df["Symbol"] != "END"].copy()
        _master_df["date_yyyymmdd"] = int(_date_yyyymmdd)
        _master_df = _master_df.rename(
            columns={
                "Symbol": "symbol",
                "Round_Lot": "round_lot",
            },
        )
        _master_df["round_lot"] = _master_df["round_lot"].astype("Int64")
        _master_frames.append(_master_df)

    master_round_lot_df = pd.concat(_master_frames, ignore_index=True)
    _selected_pair_with_round_lot_df = _selected_pair_df.merge(
        master_round_lot_df,
        on=["date_yyyymmdd", "symbol"],
        how="left",
    )

    _left_df = _selected_pair_with_round_lot_df.loc[
        _selected_pair_with_round_lot_df["date_yyyymmdd"] == left_date,
        ["symbol", mode_column, "round_lot", *cross_k_columns],
    ].copy()
    _left_df = _left_df.rename(
        columns={
            mode_column: f"mode_{left_date}",
            "round_lot": f"round_lot_{left_date}",
            **{column_name: f"{column_name}_{left_date}" for column_name in cross_k_columns},
        },
    )

    _right_df = _selected_pair_with_round_lot_df.loc[
        _selected_pair_with_round_lot_df["date_yyyymmdd"] == right_date,
        ["symbol", mode_column, "round_lot", *cross_k_columns],
    ].copy()
    _right_df = _right_df.rename(
        columns={
            mode_column: f"mode_{right_date}",
            "round_lot": f"round_lot_{right_date}",
            **{column_name: f"{column_name}_{right_date}" for column_name in cross_k_columns},
        },
    )

    paired_mode_change_df = _left_df.merge(_right_df, on="symbol", how="inner")
    paired_mode_change_df["mode_change"] = (
        paired_mode_change_df[f"mode_{right_date}"] - paired_mode_change_df[f"mode_{left_date}"]
    )
    paired_mode_change_df["round_lot_changed"] = (
        paired_mode_change_df[f"round_lot_{left_date}"]
        != paired_mode_change_df[f"round_lot_{right_date}"]
    )
    for _column_name in cross_k_columns:
        paired_mode_change_df[f"{_column_name}_change"] = (
            paired_mode_change_df[f"{_column_name}_{right_date}"]
            - paired_mode_change_df[f"{_column_name}_{left_date}"]
        )
    return (paired_mode_change_df,)


@app.cell
def _(cta_glob, glob, pd):
    _cta_paths = sorted(path for path in glob.glob(cta_glob))
    _cta_frames = [
        pd.read_csv(
            _path,
            usecols=["Symbol", "PrimaryListingMarketPreviousClosingPrice"],
        )
        for _path in _cta_paths
    ]
    mean_close_202509_df = (
        pd.concat(_cta_frames, ignore_index=True)
        .groupby("Symbol", as_index=False)["PrimaryListingMarketPreviousClosingPrice"]
        .mean()
        .rename(
            columns={
                "Symbol": "symbol",
                "PrimaryListingMarketPreviousClosingPrice": "mean_close_202509",
            },
        )
        .sort_values("symbol", kind="stable")
    )
    return (mean_close_202509_df,)


@app.cell
def _(
    cross_k_columns,
    left_date,
    mean_close_202509_df,
    paired_mode_change_df,
    right_date,
):
    paired_mode_change_with_mean_close_df = paired_mode_change_df.merge(
        mean_close_202509_df,
        on="symbol",
        how="left",
    )
    ordered_columns = [
        "symbol",
        f"mode_{left_date}",
        f"mode_{right_date}",
        "mode_change",
        f"round_lot_{left_date}",
        f"round_lot_{right_date}",
        "round_lot_changed",
        *[
            column_name
            for _base_name in cross_k_columns
            for column_name in (
                f"{_base_name}_{left_date}",
                f"{_base_name}_{right_date}",
                f"{_base_name}_change",
            )
        ],
        "mean_close_202509",
    ]
    paired_mode_change_with_mean_close_df = (
        paired_mode_change_with_mean_close_df.loc[:, ordered_columns]
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )
    paired_mode_change_with_mean_close_df
    return (paired_mode_change_with_mean_close_df,)


if __name__ == "__main__":
    app.run()
