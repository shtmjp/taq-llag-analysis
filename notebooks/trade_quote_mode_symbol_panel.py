import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import pandas as pd
    import yfinance as yf

    return Path, json, pd, yf


@app.cell
def _(Path):
    results_base_dir = Path("data/derived/trade_quote_modes")
    master_base_dir = Path("data/dailyTAQ/MASTER")
    output_base_dir = Path("data/derived/trade_quote_mode_symbol_panel")
    run_id = ""
    trade_exchange = "Q"
    quote_exchange = "Z"
    mode_column = "closest_mode_to_zero_sec"
    start = "2025-09-01"
    end = "2025-10-01"
    output_csv_path = None
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
        end,
        master_base_dir,
        mode_column,
        output_base_dir,
        output_csv_path,
        quote_exchange,
        results_base_dir,
        run_id,
        start,
        trade_exchange,
    )


@app.cell
def _(json, results_base_dir, run_id):
    available_run_ids = (
        sorted(path.name for path in results_base_dir.iterdir() if path.is_dir())
        if results_base_dir.exists()
        else []
    )
    if run_id:
        resolved_run_id = run_id
    elif available_run_ids:
        resolved_run_id = available_run_ids[-1]
    else:
        message = f"No run directories found under {results_base_dir}."
        raise FileNotFoundError(message)

    run_dir = results_base_dir / resolved_run_id
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    pair_summary_path = run_dir / "pair_summary.csv"
    target_dates = tuple(str(date_yyyymmdd) for date_yyyymmdd in run_config["target_dates"])
    return pair_summary_path, resolved_run_id, target_dates


@app.cell
def _(
    Path,
    cross_k_columns,
    end,
    master_base_dir,
    mode_column,
    output_base_dir,
    output_csv_path,
    pair_summary_path,
    pd,
    quote_exchange,
    resolved_run_id,
    start,
    target_dates,
    trade_exchange,
    yf,
):
    def normalize_for_yahoo(symbol: str) -> str:
        return str(symbol).strip().upper().replace(".", "-").replace("/", "-")

    pair_summary_df = pd.read_csv(pair_summary_path)
    pair_summary_df["date_yyyymmdd"] = pair_summary_df["date_yyyymmdd"].astype(str)
    selected_pair_df = pair_summary_df.loc[
        (pair_summary_df["status"] == "ok")
        & (pair_summary_df["trade_exchange"] == trade_exchange)
        & (pair_summary_df["quote_exchange"] == quote_exchange),
        ["symbol", "date_yyyymmdd", mode_column, *cross_k_columns],
    ].copy()
    selected_pair_df = selected_pair_df.rename(columns={mode_column: "mode"})

    master_frames = []
    for date_yyyymmdd in target_dates:
        master_df = pd.read_csv(
            master_base_dir / f"EQY_US_ALL_REF_MASTER_{date_yyyymmdd}",
            sep="|",
            encoding="latin1",
            usecols=["Symbol", "Round_Lot"],
        )
        master_df = master_df.loc[master_df["Symbol"] != "END"].copy()
        master_df["date_yyyymmdd"] = date_yyyymmdd
        master_df = master_df.rename(
            columns={
                "Symbol": "symbol",
                "Round_Lot": "round_lot",
            },
        )
        master_frames.append(master_df)

    master_round_lot_df = pd.concat(master_frames, ignore_index=True)
    panel_source_df = selected_pair_df.merge(
        master_round_lot_df,
        on=["symbol", "date_yyyymmdd"],
        how="left",
    )

    value_columns = ["round_lot", "mode", *cross_k_columns]
    wide_parts = []
    for column_name in value_columns:
        wide_part = panel_source_df.pivot_table(
            index="symbol",
            columns="date_yyyymmdd",
            values=column_name,
            aggfunc="first",
        )
        wide_part = wide_part.reindex(columns=list(target_dates))
        wide_part = wide_part.rename(
            columns={
                date_yyyymmdd: f"{column_name}_{date_yyyymmdd}" for date_yyyymmdd in target_dates
            },
        )
        wide_parts.append(wide_part)

    wide_df = pd.concat(wide_parts, axis=1)
    wide_df.index.name = "symbol"
    wide_df = wide_df.sort_index().reset_index()

    yahoo_symbol_df = wide_df.loc[:, ["symbol"]].copy()
    yahoo_symbol_df["symbol_yahoo"] = yahoo_symbol_df["symbol"].map(normalize_for_yahoo)
    symbol_yahoo_list = yahoo_symbol_df["symbol_yahoo"].drop_duplicates().tolist()

    if symbol_yahoo_list:
        close_data = yf.download(
            tickers=symbol_yahoo_list,
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
        if close_data is None or close_data.empty:
            mean_close_by_yahoo_df = pd.DataFrame(
                {
                    "symbol_yahoo": symbol_yahoo_list,
                    "mean_close_202509": pd.NA,
                },
            )
        else:
            if not isinstance(close_data.columns, pd.MultiIndex):
                close_data = pd.concat({symbol_yahoo_list[0]: close_data}, axis=1)

            close_frames = [
                close_data[symbol_yahoo][["Close"]].rename(columns={"Close": symbol_yahoo})
                for symbol_yahoo in symbol_yahoo_list
                if symbol_yahoo in set(close_data.columns.get_level_values(0))
            ]
            if close_frames:
                daily_close_df = pd.concat(close_frames, axis=1)
                mean_close = daily_close_df.mean(axis=0, skipna=True)
                mean_close_by_yahoo_df = pd.DataFrame(
                    {
                        "symbol_yahoo": mean_close.index.astype(str),
                        "mean_close_202509": mean_close.to_numpy(),
                    },
                )
            else:
                mean_close_by_yahoo_df = pd.DataFrame(
                    columns=["symbol_yahoo", "mean_close_202509"],
                )
    else:
        mean_close_by_yahoo_df = pd.DataFrame(columns=["symbol_yahoo", "mean_close_202509"])

    mean_close_202509_df = yahoo_symbol_df.merge(
        mean_close_by_yahoo_df,
        on="symbol_yahoo",
        how="left",
    ).loc[:, ["symbol", "mean_close_202509"]]

    final_df = wide_df.merge(mean_close_202509_df, on="symbol", how="left")

    ordered_columns = ["symbol", "mean_close_202509"]
    for date_yyyymmdd in target_dates:
        ordered_columns.extend(
            [
                f"round_lot_{date_yyyymmdd}",
                f"mode_{date_yyyymmdd}",
                *[f"{column_name}_{date_yyyymmdd}" for column_name in cross_k_columns],
            ],
        )
    final_df = (
        final_df.loc[:, ordered_columns]
        .sort_values("symbol", kind="stable")
        .reset_index(
            drop=True,
        )
    )

    resolved_output_csv_path = (
        Path(output_csv_path)
        if output_csv_path is not None
        else output_base_dir
        / (
            f"{resolved_run_id}_{trade_exchange}{quote_exchange}"
            "_with_round_lot_sep2025_mean_close.csv"
        )
    )
    resolved_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(resolved_output_csv_path, index=False)
    return (final_df,)


@app.cell
def _(final_df):
    for _r in [10, 40, 100]:
        print(final_df.dropna()[final_df.dropna()["round_lot_20251103"] == _r].head(5))
    return


if __name__ == "__main__":
    app.run()
