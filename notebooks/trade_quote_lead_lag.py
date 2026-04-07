import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import ppllag

    return Path, np, pl, plt, ppllag


@app.cell
def _(Path, np):
    trade_base_dir = Path("data/filtered/trade")
    quote_base_dir = Path("data/filtered/quote")
    date_yyyymmdd = "20251103"
    symbol = "MELI"
    trade_exchange = "Q"
    quote_exchange = "Z"
    obs_window = (35_100.0, 56_701.0)
    u_range = (-1.0, 1.0)
    bw_candidates = np.array([1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
    u_grid = np.linspace(-1e-3, 1e-3, 2001, dtype=np.float64)
    kernel = "tent"
    allow_not_simple = False
    trade_path = trade_base_dir / symbol / f"trade_{date_yyyymmdd}.parquet"
    quote_path = quote_base_dir / symbol / f"quote_{date_yyyymmdd}.parquet"
    return (
        allow_not_simple,
        bw_candidates,
        date_yyyymmdd,
        kernel,
        obs_window,
        quote_exchange,
        quote_path,
        symbol,
        trade_exchange,
        trade_path,
        u_grid,
        u_range,
    )


@app.cell
def _(
    date_yyyymmdd,
    pl,
    quote_exchange,
    quote_path,
    symbol,
    trade_exchange,
    trade_path,
):
    input_status_df = pl.DataFrame(
        {
            "dataset": ["trade", "quote"],
            "date_yyyymmdd": [date_yyyymmdd, date_yyyymmdd],
            "symbol": [symbol, symbol],
            "exchange": [trade_exchange, quote_exchange],
            "path": [str(trade_path), str(quote_path)],
            "exists": [trade_path.exists(), quote_path.exists()],
        },
    )
    input_status_df
    return


@app.cell
def _(quote_path, trade_path):
    missing_paths = [str(path) for path in (trade_path, quote_path) if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(missing_paths)
        raise FileNotFoundError(
            f"Missing filtered parquet file(s): {missing_display}",
        )
    return


@app.cell
def _(pl):
    def timestamp_to_time_expr(timestamp_col: str) -> pl.Expr:
        timestamp = pl.col(timestamp_col)
        hh = (timestamp // 10_000_000_000_000).cast(pl.Float64)
        mm = ((timestamp // 100_000_000_000) % 100).cast(pl.Float64)
        ss = ((timestamp // 1_000_000_000) % 100).cast(pl.Float64)
        subsec = (timestamp % 1_000_000_000).cast(pl.Float64) / 1_000_000_000.0
        return (hh * 3600.0 + mm * 60.0 + ss + subsec).alias("event_time")

    return (timestamp_to_time_expr,)


@app.cell
def _(pl, trade_path):
    trade_df = pl.read_parquet(trade_path)
    return (trade_df,)


@app.cell
def _(pl, quote_path):
    quote_df = pl.read_parquet(quote_path)
    return (quote_df,)


@app.cell
def _(pl, trade_df):
    trade_exchange_summary_df = (
        trade_df.group_by("Exchange")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("Participant Timestamp").n_unique().alias("n_events"),
        )
        .sort("n_events", descending=True)
    )
    trade_exchange_summary_df
    return (trade_exchange_summary_df,)


@app.cell
def _(pl, quote_df):
    quote_exchange_summary_df = (
        quote_df.group_by("Exchange")
        .agg(
            pl.len().alias("n_rows"),
            pl.col("Participant_Timestamp").n_unique().alias("n_events"),
        )
        .sort("n_events", descending=True)
    )
    quote_exchange_summary_df
    return (quote_exchange_summary_df,)


@app.cell
def _(
    np,
    pl,
    timestamp_to_time_expr,
    trade_df,
    trade_exchange,
    trade_exchange_summary_df,
):
    available_trade_exchanges = trade_exchange_summary_df.get_column("Exchange").to_list()
    if trade_exchange not in available_trade_exchanges:
        _available_display = ", ".join(available_trade_exchanges)
        raise ValueError(
            "Selected trade exchange is unavailable: "
            f"{trade_exchange}. Available trade exchanges: {_available_display}",
        )

    trade_events_df = (
        trade_df.filter(pl.col("Exchange") == trade_exchange)
        .group_by(["Exchange", "Participant Timestamp"])
        .agg(pl.len().alias("n_rows"))
        .with_columns(timestamp_to_time_expr("Participant Timestamp"))
        .select(["Exchange", "Participant Timestamp", "n_rows", "event_time"])
        .sort("event_time")
    )
    if trade_events_df.height == 0:
        raise ValueError(
            f"No trade events remain after collapse for exchange {trade_exchange}.",
        )

    trade_event_times = np.asarray(
        trade_events_df.get_column("event_time").to_numpy(),
        dtype=np.float64,
    )
    return (trade_event_times,)


@app.cell
def _(
    np,
    pl,
    quote_df,
    quote_exchange,
    quote_exchange_summary_df,
    timestamp_to_time_expr,
):
    available_quote_exchanges = quote_exchange_summary_df.get_column("Exchange").to_list()
    if quote_exchange not in available_quote_exchanges:
        _available_display = ", ".join(available_quote_exchanges)
        raise ValueError(
            "Selected quote exchange is unavailable: "
            f"{quote_exchange}. Available quote exchanges: {_available_display}",
        )

    quote_events_df = (
        quote_df.filter(pl.col("Exchange") == quote_exchange)
        .group_by(["Exchange", "Participant_Timestamp"])
        .agg(pl.len().alias("n_rows"))
        .with_columns(timestamp_to_time_expr("Participant_Timestamp"))
        .select(["Exchange", "Participant_Timestamp", "n_rows", "event_time"])
        .sort("event_time")
    )
    if quote_events_df.height == 0:
        raise ValueError(
            f"No quote events remain after collapse for exchange {quote_exchange}.",
        )

    quote_event_times = np.asarray(
        quote_events_df.get_column("event_time").to_numpy(),
        dtype=np.float64,
    )
    return (quote_event_times,)


@app.cell
def _(
    date_yyyymmdd,
    pl,
    quote_event_times,
    quote_exchange,
    quote_path,
    symbol,
    trade_event_times,
    trade_exchange,
    trade_path,
):
    pair_summary_df = pl.DataFrame(
        {
            "date_yyyymmdd": [date_yyyymmdd],
            "symbol": [symbol],
            "trade_exchange": [trade_exchange],
            "quote_exchange": [quote_exchange],
            "n_trade_events": [int(trade_event_times.size)],
            "n_quote_events": [int(quote_event_times.size)],
            "trade_path": [str(trade_path)],
            "quote_path": [str(quote_path)],
        },
    )
    pair_summary_df
    return


@app.cell
def _(date_yyyymmdd, quote_exchange, symbol, trade_exchange):
    interpretation = (
        f"Configured analysis: {date_yyyymmdd} {symbol}, "
        f"{trade_exchange} trade -> {quote_exchange} quote."
    )
    interpretation
    return


@app.cell
def _(
    allow_not_simple,
    bw_candidates,
    kernel,
    obs_window,
    ppllag,
    quote_event_times,
    trade_event_times,
    u_range,
):
    bandwidth = ppllag.lepski_bw_selector_for_cpcf_mode(
        trade_event_times,
        quote_event_times,
        obs_window=obs_window,
        u_range=u_range,
        bw_candidates=bw_candidates.tolist(),
        kernel=kernel,
        allow_not_simple=allow_not_simple,
    )
    bandwidth
    return (bandwidth,)


@app.cell
def _(
    allow_not_simple,
    bandwidth,
    kernel,
    obs_window,
    ppllag,
    quote_event_times,
    trade_event_times,
    u_grid,
    u_range,
):
    cpcf_values = ppllag.cpcf(
        trade_event_times,
        quote_event_times,
        u_values=u_grid,
        obs_window=obs_window,
        bandwidth=bandwidth,
        kernel=kernel,
        allow_not_simple=allow_not_simple,
    )
    modes = ppllag.find_cpcf_modes(
        trade_event_times,
        quote_event_times,
        u_range=u_range,
        obs_window=obs_window,
        bandwidth=bandwidth,
        kernel=kernel,
        allow_not_simple=allow_not_simple,
    )
    return cpcf_values, modes


@app.cell
def _(modes):
    modes
    return


@app.cell
def _(modes, pl):
    if modes.size == 0:
        modes_df = pl.DataFrame(schema={"mode_sec": pl.Float64})
    else:
        modes_df = pl.DataFrame({"mode_sec": modes.tolist()})
    modes_df
    return


@app.cell
def _(
    bandwidth,
    cpcf_values,
    date_yyyymmdd,
    modes,
    np,
    plt,
    quote_exchange,
    symbol,
    trade_exchange,
    u_grid,
):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(u_grid, cpcf_values, color="#0f766e", linewidth=1.5)
    ax.axvline(0.0, color="#57534e", linestyle="--", linewidth=1.0)
    if modes.size > 0:
        mode_values = np.interp(modes, u_grid, cpcf_values)
        ax.scatter(modes, mode_values, color="#b91c1c", s=28, zorder=3)
    ax.set_xlabel("Lag u (seconds)")
    ax.set_ylabel("CPCF")
    ax.set_title(
        f"{symbol} CPCF ({trade_exchange} trade -> {quote_exchange} quote), "
        f"{date_yyyymmdd}, bandwidth={bandwidth:.4g}",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
