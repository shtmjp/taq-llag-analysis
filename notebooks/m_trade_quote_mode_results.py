import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl

    return Path, json, pl, plt


@app.cell
def _(Path):
    results_base_dir = Path("data/derived/m_trade_quote_modes")
    run_id = ""
    cross_k_columns = [
        "cross_k_neg_1e3_1e4",
        "cross_k_neg_1e4_1e5",
        "cross_k_neg_1e5_0",
        "cross_k_pos_0_1e5",
        "cross_k_pos_1e5_1e4",
        "cross_k_pos_1e4_1e3",
    ]
    return cross_k_columns, results_base_dir, run_id


@app.cell
def _(results_base_dir):
    available_run_ids = (
        sorted(path.name for path in results_base_dir.iterdir() if path.is_dir())
        if results_base_dir.exists()
        else []
    )
    available_run_ids
    return (available_run_ids,)


@app.cell
def _(available_run_ids, results_base_dir, run_id):
    if run_id:
        resolved_run_id = run_id
    elif available_run_ids:
        resolved_run_id = available_run_ids[-1]
    else:
        raise FileNotFoundError(
            f"No run directories found under {results_base_dir}.",
        )

    run_dir = results_base_dir / resolved_run_id
    run_files = {
        "run_config_json": run_dir / "run_config.json",
        "symbol_inventory_csv": run_dir / "symbol_inventory.csv",
        "pair_summary_csv": run_dir / "pair_summary.csv",
        "modes_csv": run_dir / "modes.csv",
    }
    run_file_status_rows = [
        {
            "artifact": artifact_name,
            "path": str(path),
            "exists": path.exists(),
        }
        for artifact_name, path in run_files.items()
    ]
    return resolved_run_id, run_dir, run_file_status_rows, run_files


@app.cell
def _(pl, run_file_status_rows):
    file_status_df = pl.DataFrame(run_file_status_rows)
    file_status_df
    return (file_status_df,)


@app.cell
def _(file_status_df, pl):
    missing_artifacts = file_status_df.filter(~pl.col("exists"))
    if missing_artifacts.height > 0:
        missing_paths = missing_artifacts.get_column("path").to_list()
        missing_display = ", ".join(missing_paths)
        raise FileNotFoundError(f"Missing run artifacts: {missing_display}")
    return


@app.cell
def _(json, run_files):
    run_config = json.loads(run_files["run_config_json"].read_text(encoding="utf-8"))
    run_config
    return (run_config,)


@app.cell
def _(pl, run_files):
    symbol_inventory_df = pl.read_csv(run_files["symbol_inventory_csv"])
    pair_summary_df = pl.read_csv(run_files["pair_summary_csv"])
    modes_df = pl.read_csv(run_files["modes_csv"])
    return modes_df, pair_summary_df, symbol_inventory_df


@app.cell
def _(pl, symbol_inventory_df):
    inventory_summary_df = (
        symbol_inventory_df.group_by("date_yyyymmdd")
        .agg(
            pl.len().alias("n_symbol_rows"),
            pl.col("n_candidate_pairs").sum().alias("n_candidate_pairs"),
            pl.col("n_pairs_ge_min_events").sum().alias("n_pairs_ge_min_events"),
        )
        .sort("date_yyyymmdd")
    )
    inventory_summary_df
    return (inventory_summary_df,)


@app.cell
def _(pair_summary_df, pl):
    status_summary_df = (
        pair_summary_df.group_by("status")
        .agg(pl.len().alias("n_pairs"))
        .sort("n_pairs", descending=True)
    )
    status_summary_df
    return (status_summary_df,)


@app.cell
def _(pair_summary_df, pl):
    per_date_status_df = (
        pair_summary_df.group_by(["date_yyyymmdd", "status"])
        .agg(pl.len().alias("n_pairs"))
        .sort(["date_yyyymmdd", "status"])
    )
    per_date_status_df
    return (per_date_status_df,)


@app.cell
def _(pair_summary_df, pl):
    per_symbol_status_df = (
        pair_summary_df.group_by(["date_yyyymmdd", "symbol", "status"])
        .agg(pl.len().alias("n_pairs"))
        .sort(["date_yyyymmdd", "symbol", "status"])
    )
    per_symbol_status_df
    return (per_symbol_status_df,)


@app.cell
def _(pair_summary_df, pl):
    ok_pairs_df = pair_summary_df.filter(pl.col("status") == "ok")
    ok_pairs_df
    return (ok_pairs_df,)


@app.cell
def _(pair_summary_df, pl):
    error_summary_df = (
        pair_summary_df.filter(pl.col("status") == "error")
        .group_by("error_type")
        .agg(pl.len().alias("n_pairs"))
        .sort("n_pairs", descending=True)
    )
    error_summary_df
    return (error_summary_df,)


@app.cell
def _(ok_pairs_df, pl):
    mode_count_summary_df = (
        ok_pairs_df.group_by("mode_count").agg(pl.len().alias("n_pairs")).sort("mode_count")
    )
    mode_count_summary_df
    return (mode_count_summary_df,)


@app.cell
def _(cross_k_columns, ok_pairs_df):
    cross_k_summary_df = ok_pairs_df.select(cross_k_columns).describe()
    cross_k_summary_df
    return (cross_k_summary_df,)


@app.cell
def _(modes_df, pl):
    modes_per_date_df = (
        modes_df.group_by("date_yyyymmdd").agg(pl.len().alias("n_modes")).sort("date_yyyymmdd")
    )
    modes_per_date_df
    return (modes_per_date_df,)


@app.cell
def _(ok_pairs_df, plt):
    _fig, ax = plt.subplots(figsize=(9, 4.5))
    if ok_pairs_df.height > 0:
        ax.hist(
            ok_pairs_df.get_column("closest_mode_to_zero_sec").drop_nulls().to_list(),
            bins=60,
            color="#0f766e",
            alpha=0.85,
        )
    ax.set_xlabel("Closest mode to zero (seconds)")
    ax.set_ylabel("Pair count")
    ax.set_title("Distribution of closest CPCF mode to zero")
    ax.grid(alpha=0.2)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(cross_k_columns, ok_pairs_df, plt):
    _fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, column_name in zip(axes.flat, cross_k_columns, strict=True):
        if ok_pairs_df.height > 0:
            axis.hist(
                ok_pairs_df.get_column(column_name).drop_nulls().to_list(),
                bins=60,
                color="#1d4ed8",
                alpha=0.8,
            )
        axis.set_title(column_name)
        axis.grid(alpha=0.2)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
