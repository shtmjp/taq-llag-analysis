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

    import sys

    sys.path.append("../src")

    from daily_taq_trade_filter import filtered_trade_lazy_frame

    return Path, filtered_trade_lazy_frame, np, pl, plt, ppllag


@app.cell
def _(Path, np):
    input_path = Path("../data/dailyTAQ/TRADE/EQY_US_ALL_TRADE_20251103")
    output_path = Path(
        "../data/derived/daily_taq_trade_parquet/EQY_US_ALL_TRADE_20251103.AAPL.NQ.filtered.parquet",
    )
    symbol = "AAPL"
    exchanges = ["N", "Q"]
    market_open_sec = 35_100
    market_close_sec = 56_700
    obs_window = (35_100.0, 56_701.0)
    u_range = (-1.0, 1.0)
    bw_candidates = np.array([1e-7, 1e-6, 1e-5, 1e-4], dtype=np.float64)
    kernel = "tent"
    allow_not_simple = True
    u_values = np.linspace(-1e-3, 1e-3, 2001, dtype=np.float64)
    return (
        allow_not_simple,
        bw_candidates,
        input_path,
        kernel,
        market_close_sec,
        market_open_sec,
        obs_window,
        output_path,
        symbol,
        u_range,
    )


@app.cell
def _(
    filtered_trade_lazy_frame,
    input_path,
    market_close_sec,
    market_open_sec,
    output_path,
    symbol,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        filtered_trade_lazy_frame(
            input_path,
            symbols=[symbol],
            exchanges=None,
            market_open_sec=market_open_sec,
            market_close_sec=market_close_sec,
            all_columns=True,
            add_time_columns=True,
        ).sink_parquet(output_path, statistics=False)

    parquet_path = output_path
    return (parquet_path,)


@app.cell
def _(parquet_path, pl):
    trade_df = pl.read_parquet(parquet_path).sort(
        ["Exchange", "participant_timestamp_time"],
    )
    event_summary = trade_df.group_by("Exchange").agg(pl.len().alias("n_trades")).sort("Exchange")
    event_summary
    return (trade_df,)


@app.cell
def _(np, pl, trade_df):
    data1 = np.asarray(
        trade_df.filter(pl.col("Exchange") == "Q")
        .get_column("participant_timestamp_time")
        .to_numpy(),
        dtype=np.float64,
    )
    data2 = np.asarray(
        trade_df.filter(pl.col("Exchange") == "Z")
        .get_column("participant_timestamp_time")
        .to_numpy(),
        dtype=np.float64,
    )
    return data1, data2


@app.cell
def _(
    allow_not_simple,
    bw_candidates,
    data1,
    data2,
    kernel,
    obs_window,
    ppllag,
    u_range,
):
    bandwidth = ppllag.lepski_bw_selector_for_cpcf_mode(
        data1,
        data2,
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
    data1,
    data2,
    kernel,
    np,
    obs_window,
    ppllag,
    u_range,
):
    u_grid = np.linspace(-1e-3, 1e-3, 2001, dtype=np.float64)

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
    modes
    return cpcf_values, modes, u_grid


@app.cell
def _(bandwidth, cpcf_values, modes, np, plt, symbol, u_grid):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(u_grid, cpcf_values, color="#0f766e", linewidth=1.5)
    ax.axvline(0.0, color="#57534e", linestyle="--", linewidth=1.0)
    if modes.size > 0:
        mode_values = np.interp(modes, u_grid, cpcf_values)
        ax.scatter(modes, mode_values, color="#b91c1c", s=28, zorder=3)
    ax.set_xlabel("Lag u (seconds)")
    ax.set_ylabel("CPCF")
    ax.set_title(f"{symbol} CPCF (N -> Q), bandwidth={bandwidth:.4g}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig
    return


@app.cell
def _(data1, np, obs_window, plt, ppllag, symbol):
    _grid = np.linspace(0, 1e-4, 1001, dtype=np.float64)

    pcf_values1 = ppllag.pcf(
        np.unique(np.sort(data1)),
        u_values=_grid,
        obs_window=obs_window,
        bandwidth=1e-6,
        allow_not_simple=True,
    )

    plt.plot(_grid, pcf_values1, color="#0f766e", linewidth=1.5)
    plt.title(f"{symbol} PCF (Q), bandwidth=1e-6")
    return


@app.cell
def _(data2, np, obs_window, plt, ppllag, symbol):
    _grid = np.linspace(0, 1e-4, 1001, dtype=np.float64)

    pcf_values2 = ppllag.pcf(
        np.unique(np.sort(data2)),
        u_values=_grid,
        obs_window=obs_window,
        bandwidth=1e-5,
        allow_not_simple=True,
    )

    plt.plot(_grid, pcf_values2, color="#0f766e", linewidth=1.5)
    plt.title(f"{symbol} PCF (Z), bandwidth=1e-6")
    return


if __name__ == "__main__":
    app.run()
