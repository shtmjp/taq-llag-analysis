import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    from pathlib import Path

    return (pl,)


@app.cell
def _(pl):
    mean_close_202509_df = (
        pl.scan_csv(
            str("data/cta_symbol_files_202509/CTA.Symbol.File.202509*.csv")
        )
        .group_by("Symbol")   # 列名は実際のCSVに合わせて変更
        .agg(
            pl.col("PrimaryListingMarketPreviousClosingPrice").mean().alias("mean_close_202509")
        )
        .sort("Symbol")
        .collect()
    )
    return (mean_close_202509_df,)


@app.cell
def _(mean_close_202509_df, pl):
    master_20251103_df = pl.read_csv(
        "data/dailyTAQ/MASTER/EQY_US_ALL_REF_MASTER_20251103",
        separator="|",
        has_header=True,
        comment_prefix="END",
        encoding="latin1",
    )
    master_20251103_df = master_20251103_df.join(mean_close_202509_df, on="Symbol")
    return (master_20251103_df,)


@app.cell
def _(master_20251103_df, pl):
    master_20251103_df.filter(
        (pl.col("Round_Lot") == 40) !=
        (pl.col("mean_close_202509").is_between(250, 1000))
    )
    return


@app.cell
def _(master_20251103_df, pl):
    master_20251103_df.filter(
        (pl.col("Round_Lot") == 40) !=
        (pl.col("mean_close_202509").is_between(250, 1000, closed="left"))
    )
    return


@app.cell
def _(master_20251103_df, pl):
    master_20251103_df.filter(
        (pl.col("Round_Lot") == 10) !=
        (pl.col("mean_close_202509").is_between(1000, 10000, closed="left"))
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
