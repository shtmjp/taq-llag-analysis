import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import sys

    import polars as pl

    src_path = str(Path("src").resolve())
    if src_path not in sys.path:
        sys.path.append(src_path)

    from write_filtered_trade_parquet import write_filtered_trade_parquets

    return Path, pl, write_filtered_trade_parquets


@app.cell
def _(Path):
    master_path = Path("data/dailyTAQ/MASTER/EQY_US_ALL_REF_MASTER_20251103")
    target_dates = ["20251103", "20251030"]
    return master_path, target_dates


@app.cell
def _(master_path, pl):
    symbols = (
        pl.read_csv(
            master_path,
            separator="|",
            has_header=True,
            comment_prefix="END",
            encoding="latin1",
        )
        .filter(pl.col("Round_Lot") == 10)
        .select("Symbol")
        .sort("Symbol")
        .get_column("Symbol")
        .to_list()
    )
    return (symbols,)


@app.cell
def _(symbols, target_dates, write_filtered_trade_parquets):
    written_paths_by_date = {
        date_yyyymmdd: write_filtered_trade_parquets(symbols, date_yyyymmdd)
        for date_yyyymmdd in target_dates
    }
    return (written_paths_by_date,)


@app.cell
def _(pl, written_paths_by_date):
    summary_rows = [
        {
            "date_yyyymmdd": date_yyyymmdd,
            "Symbol": symbol,
            "output_path": str(path),
            "size_bytes": path.stat().st_size,
        }
        for date_yyyymmdd, written_paths in written_paths_by_date.items()
        for symbol, path in written_paths.items()
    ]
    summary_df = pl.DataFrame(summary_rows).sort(["date_yyyymmdd", "Symbol"])
    summary_df
    return (summary_df,)


if __name__ == "__main__":
    app.run()
