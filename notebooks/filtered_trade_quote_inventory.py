import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import re

    import marimo as mo
    import pandas as pd
    import pyarrow.parquet as pq

    mb_divisor = 1024**2
    repo_root = Path(__file__).resolve().parents[1]
    trade_dir = repo_root / "data" / "filtered" / "trade"
    quote_dir = repo_root / "data" / "filtered" / "quote"
    return Path, mb_divisor, mo, pd, pq, quote_dir, re, repo_root, trade_dir


@app.cell
def _(mo):
    overview = mo.md(
        """
        ## Overview

        This notebook inventories all filtered trade and quote parquet files.
        It reports row counts from parquet metadata, file sizes, dataset-level
        totals, and symbol-level aggregates without reading parquet bodies.
        """
    )
    overview
    return


@app.cell
def _():
    top_n = 100
    return (top_n,)


@app.cell
def _(mb_divisor, pd, pq, re, repo_root):
    def extract_date_yyyymmdd(file_path):
        """Extract the YYYYMMDD token from a parquet filename."""
        matched = re.search(r"(\d{8})", file_path.stem)
        return matched.group(1)

    def build_file_inventory(*, dataset, dataset_dir):
        """Scan one filtered dataset directory into a file-level dataframe."""
        records = []
        for file_path in sorted(dataset_dir.glob("*/*.parquet")):
            n_rows = pq.ParquetFile(file_path).metadata.num_rows
            file_size_bytes = file_path.stat().st_size
            try:
                display_path = str(file_path.relative_to(repo_root))
            except ValueError:
                display_path = str(file_path)
            records.append(
                {
                    "dataset": dataset,
                    "symbol": file_path.parent.name,
                    "date_yyyymmdd": extract_date_yyyymmdd(file_path),
                    "path": display_path,
                    "n_rows": int(n_rows),
                    "file_size_bytes": int(file_size_bytes),
                    "file_size_mb": file_size_bytes / mb_divisor,
                }
            )

        file_inventory_df = pd.DataFrame.from_records(
            records,
            columns=[
                "dataset",
                "symbol",
                "date_yyyymmdd",
                "path",
                "n_rows",
                "file_size_bytes",
                "file_size_mb",
            ],
        )
        if file_inventory_df.empty:
            return file_inventory_df

        return file_inventory_df.sort_values(
            by=["file_size_bytes", "n_rows", "dataset", "symbol", "date_yyyymmdd", "path"],
            ascending=[False, False, True, True, True, True],
            ignore_index=True,
        )

    def build_dataset_summary(file_inventory_df):
        """Aggregate file inventory into dataset-level totals."""
        dataset_summary_df = (
            file_inventory_df.groupby("dataset", as_index=False)
            .agg(
                n_files=("path", "count"),
                n_symbols=("symbol", "nunique"),
                n_rows_total=("n_rows", "sum"),
                file_size_bytes_total=("file_size_bytes", "sum"),
            )
            .assign(
                file_size_mb_total=lambda frame: frame["file_size_bytes_total"] / mb_divisor,
            )
        )
        dataset_summary_df["dataset"] = pd.Categorical(
            dataset_summary_df["dataset"],
            categories=["trade", "quote"],
            ordered=True,
        )
        dataset_summary_df = dataset_summary_df.sort_values(
            by="dataset",
            ignore_index=True,
        )
        dataset_summary_df["dataset"] = dataset_summary_df["dataset"].astype(str)
        return dataset_summary_df[
            [
                "dataset",
                "n_files",
                "n_symbols",
                "n_rows_total",
                "file_size_bytes_total",
                "file_size_mb_total",
            ]
        ]

    def build_symbol_summary(file_inventory_df):
        """Aggregate file inventory into symbol-level totals."""
        trade_summary_df = (
            file_inventory_df.loc[file_inventory_df["dataset"] == "trade"]
            .groupby("symbol", as_index=False)
            .agg(
                trade_file_count=("path", "count"),
                trade_rows_total=("n_rows", "sum"),
                trade_size_bytes_total=("file_size_bytes", "sum"),
            )
        )
        quote_summary_df = (
            file_inventory_df.loc[file_inventory_df["dataset"] == "quote"]
            .groupby("symbol", as_index=False)
            .agg(
                quote_file_count=("path", "count"),
                quote_rows_total=("n_rows", "sum"),
                quote_size_bytes_total=("file_size_bytes", "sum"),
            )
        )
        symbol_summary_df = trade_summary_df.merge(
            quote_summary_df,
            on="symbol",
            how="outer",
        )
        numeric_columns = [
            "trade_file_count",
            "quote_file_count",
            "trade_rows_total",
            "quote_rows_total",
            "trade_size_bytes_total",
            "quote_size_bytes_total",
        ]
        symbol_summary_df[numeric_columns] = (
            symbol_summary_df[numeric_columns].fillna(0).astype("int64")
        )
        symbol_summary_df["total_rows"] = (
            symbol_summary_df["trade_rows_total"] + symbol_summary_df["quote_rows_total"]
        )
        symbol_summary_df["total_size_bytes"] = (
            symbol_summary_df["trade_size_bytes_total"]
            + symbol_summary_df["quote_size_bytes_total"]
        )
        symbol_summary_df["total_size_mb"] = symbol_summary_df["total_size_bytes"] / mb_divisor
        return symbol_summary_df[
            [
                "symbol",
                "trade_file_count",
                "quote_file_count",
                "trade_rows_total",
                "quote_rows_total",
                "trade_size_bytes_total",
                "quote_size_bytes_total",
                "total_rows",
                "total_size_bytes",
                "total_size_mb",
            ]
        ].sort_values(
            by=["total_size_bytes", "total_rows", "symbol"],
            ascending=[False, False, True],
            ignore_index=True,
        )

    return build_dataset_summary, build_file_inventory, build_symbol_summary


@app.cell
def _(
    build_dataset_summary,
    build_file_inventory,
    build_symbol_summary,
    pd,
    quote_dir,
    trade_dir,
):
    trade_file_inventory_df = build_file_inventory(dataset="trade", dataset_dir=trade_dir)
    quote_file_inventory_df = build_file_inventory(dataset="quote", dataset_dir=quote_dir)
    file_inventory_df = pd.concat(
        [trade_file_inventory_df, quote_file_inventory_df],
        ignore_index=True,
    ).sort_values(
        by=["file_size_bytes", "n_rows", "dataset", "symbol", "date_yyyymmdd", "path"],
        ascending=[False, False, True, True, True, True],
        ignore_index=True,
    )
    dataset_summary_df = build_dataset_summary(file_inventory_df)
    file_size_ranking_df = file_inventory_df.loc[:, :]
    trade_file_size_ranking_df = file_inventory_df.loc[
        file_inventory_df["dataset"] == "trade"
    ].reset_index(drop=True)
    quote_file_size_ranking_df = file_inventory_df.loc[
        file_inventory_df["dataset"] == "quote"
    ].reset_index(drop=True)
    symbol_summary_df = build_symbol_summary(file_inventory_df)
    return (
        dataset_summary_df,
        file_inventory_df,
        file_size_ranking_df,
        quote_file_size_ranking_df,
        symbol_summary_df,
        trade_file_size_ranking_df,
    )


@app.cell
def _(dataset_summary_df, file_inventory_df, mo, symbol_summary_df):
    scan_summary = mo.md(
        f"""
        ## Scan Summary

        Scanned `{len(file_inventory_df):,}` parquet files across
        `{dataset_summary_df["dataset"].nunique():,}` datasets and
        `{len(symbol_summary_df):,}` symbols.
        """
    )
    scan_summary
    return


@app.cell
def _(dataset_summary_df):
    dataset_summary_df
    return


@app.cell
def _(mo, top_n):
    rankings_header = mo.md(
        f"""
        ## File Size Rankings

        The tables below show the largest `{top_n}` parquet files by
        `file_size_bytes`.
        """
    )
    rankings_header
    return


@app.cell
def _(file_size_ranking_df, top_n):
    file_size_ranking_df.head(top_n)
    return


@app.cell
def _(trade_file_size_ranking_df, top_n):
    trade_file_size_ranking_df.head(top_n)
    return


@app.cell
def _(quote_file_size_ranking_df, top_n):
    quote_file_size_ranking_df.head(top_n)
    return


@app.cell
def _(mo):
    file_inventory_header = mo.md(
        """
        ## Full File Inventory

        This table includes every filtered trade and quote parquet file.
        """
    )
    file_inventory_header
    return


@app.cell
def _(file_inventory_df):
    file_inventory_df
    return


@app.cell
def _(mo):
    symbol_summary_header = mo.md(
        """
        ## Symbol Summary

        This table aggregates trade and quote file counts, row counts, and
        file sizes by symbol across all available dates.
        """
    )
    symbol_summary_header
    return


@app.cell
def _(symbol_summary_df):
    symbol_summary_df
    return


if __name__ == "__main__":
    app.run()
