import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    plt.rcParams["font.family"] = ["Hiragino Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    return UTC, Path, datetime, json, mpimg, np, pd, plt, sm


@app.cell
def _(Path):
    input_csv_path = None
    output_base_dir = Path("data/derived/m_trade_quote_round_lot_effects")
    pre_round_lot_required = 100.0
    allowed_post_round_lots = (100.0, 40.0)
    export_analysis_sample = True
    export_group_summary = True
    return (
        allowed_post_round_lots,
        export_analysis_sample,
        export_group_summary,
        input_csv_path,
        output_base_dir,
        pre_round_lot_required,
    )


@app.cell
def _(Path, input_csv_path, output_base_dir):
    panel_base_dir = Path("data/derived/trade_quote_mode_symbol_panel")
    if input_csv_path is None:
        candidate_paths = sorted(panel_base_dir.glob("*_with_round_lot_sep2025_mean_close.csv"))
        if not candidate_paths:
            message = f"No panel CSV files found under {panel_base_dir}."
            raise FileNotFoundError(message)
        resolved_input_csv_path = candidate_paths[-1]
    else:
        resolved_input_csv_path = Path(input_csv_path)

    if not resolved_input_csv_path.exists():
        message = f"Panel CSV not found: {resolved_input_csv_path}"
        raise FileNotFoundError(message)

    panel_csv_stem = resolved_input_csv_path.stem
    resolved_output_dir = output_base_dir / panel_csv_stem
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return panel_csv_stem, resolved_input_csv_path, resolved_output_dir


@app.cell
def _(Path, datetime, json, np, pd, plt, sm):  # noqa: C901, PLR0915
    DATE_PRE = "20251031"
    DATE_POST = "20251103"
    PRICE_COL = "mean_close_202509"
    PRE_RL_COL = "round_lot_20251031"
    POST_RL_COL = "round_lot_20251103"
    RULE_CUTOFF = 250.0
    TREATED_POST_ROUND_LOT = 40.0
    CONTROL_POST_ROUND_LOT = 100.0
    MIN_GROUPS = 2

    def infer_available_y_stems(columns):
        """Infer outcome stems that exist on both dates.

        Parameters
        ----------
        columns
            Column labels from the panel CSV.

        Returns
        -------
        list[str]
            Sorted outcome stems excluding `round_lot`.

        """
        pre_suffix = f"_{DATE_PRE}"
        post_suffix = f"_{DATE_POST}"
        stems = []
        for column_name in columns:
            if not str(column_name).endswith(pre_suffix):
                continue
            stem = str(column_name)[: -len(pre_suffix)]
            if stem == "round_lot":
                continue
            if f"{stem}{post_suffix}" in columns:
                stems.append(stem)
        return sorted(set(stems))

    def transform_for_outcome(y_stem):
        """Map one outcome to its comparison transform.

        Parameters
        ----------
        y_stem
            Outcome stem inferred from the panel CSV.

        Returns
        -------
        str
            `signed` for `mode`, otherwise `log1p`.

        """
        return "signed" if y_stem == "mode" else "log1p"

    def get_outcome_columns(y_stem):
        """Return the pre and post columns for an outcome.

        Parameters
        ----------
        y_stem
            Outcome stem.

        Returns
        -------
        tuple[str, str]
            Pre and post column names.

        """
        return f"{y_stem}_{DATE_PRE}", f"{y_stem}_{DATE_POST}"

    def make_delta(pre, post, transform):
        """Construct the outcome change series.

        Parameters
        ----------
        pre
            Pre-reform outcome series.
        post
            Post-reform outcome series.
        transform
            Delta transform name.

        Returns
        -------
        pandas.Series
            Change in the outcome under the configured transform.

        """
        if transform == "signed":
            return post - pre
        if transform == "log1p":
            return np.log1p(post) - np.log1p(pre)
        message = f"Unknown transform: {transform!r}"
        raise ValueError(message)

    def build_base_sample(
        raw_df,
        *,
        pre_round_lot_required,
        allowed_post_round_lots,
    ):
        """Construct the common comparison sample before outcome-specific filtering.

        Parameters
        ----------
        raw_df
            Raw panel dataframe.
        pre_round_lot_required
            Required pre-reform round lot.
        allowed_post_round_lots
            Allowed post-reform round-lot values kept in the comparison sample.

        Returns
        -------
        pandas.DataFrame
            Base sample with actual treatment assignment and diagnostic columns.

        """
        required_columns = ["symbol", PRICE_COL, PRE_RL_COL, POST_RL_COL]
        missing_columns = [
            column_name for column_name in required_columns if column_name not in raw_df.columns
        ]
        if missing_columns:
            message = f"Missing required columns: {missing_columns}"
            raise KeyError(message)

        sample_df = raw_df.copy()
        sample_df = sample_df.loc[sample_df[PRICE_COL].notna()].copy()
        sample_df = sample_df.loc[sample_df[PRE_RL_COL] == pre_round_lot_required].copy()
        sample_df = sample_df.loc[sample_df[POST_RL_COL].isin(list(allowed_post_round_lots))].copy()

        sample_df["treated"] = (sample_df[POST_RL_COL] == TREATED_POST_ROUND_LOT).astype(int)
        sample_df["group_label"] = np.where(sample_df["treated"] == 1, "to40", "stay100")
        sample_df["rule_consistent_250"] = (
            (
                (sample_df[PRICE_COL] <= RULE_CUTOFF)
                & sample_df[POST_RL_COL].eq(CONTROL_POST_ROUND_LOT)
            )
            | (
                (sample_df[PRICE_COL] > RULE_CUTOFF)
                & sample_df[POST_RL_COL].eq(TREATED_POST_ROUND_LOT)
            )
        ).astype(int)
        sample_df["actual_transition"] = (
            sample_df[PRE_RL_COL].astype("Int64").astype(str)
            + "->"
            + sample_df[POST_RL_COL].astype("Int64").astype(str)
        )
        return sample_df

    def build_outcome_sample(base_sample_df, y_stem):
        """Attach one outcome's pre, post, and delta columns to the base sample.

        Parameters
        ----------
        base_sample_df
            Common comparison sample.
        y_stem
            Outcome stem.

        Returns
        -------
        pandas.DataFrame
            Outcome-specific comparison sample.

        """
        pre_y_col, post_y_col = get_outcome_columns(y_stem)
        missing_columns = [
            column_name
            for column_name in (pre_y_col, post_y_col)
            if column_name not in base_sample_df.columns
        ]
        if missing_columns:
            message = f"Missing outcome columns: {missing_columns}"
            raise KeyError(message)

        transform = transform_for_outcome(y_stem)
        outcome_df = base_sample_df.copy()
        outcome_df["y_pre"] = pd.to_numeric(outcome_df[pre_y_col], errors="coerce")
        outcome_df["y_post"] = pd.to_numeric(outcome_df[post_y_col], errors="coerce")
        outcome_df["delta_y"] = make_delta(
            outcome_df["y_pre"],
            outcome_df["y_post"],
            transform=transform,
        )
        outcome_df = outcome_df.loc[
            outcome_df["y_pre"].notna()
            & outcome_df["y_post"].notna()
            & outcome_df["delta_y"].notna()
        ].copy()
        outcome_df["outcome"] = y_stem
        outcome_df["transform"] = transform
        return outcome_df

    def summarize_base_groups(base_sample_df):
        """Summarize the common comparison sample by actual post-reform group.

        Parameters
        ----------
        base_sample_df
            Common comparison sample.

        Returns
        -------
        pandas.DataFrame
            Group counts and price diagnostics.

        """
        return (
            base_sample_df.groupby(["group_label", "treated"], observed=True)
            .agg(
                n_obs=("symbol", "size"),
                n_symbols=("symbol", "nunique"),
                mean_price=(PRICE_COL, "mean"),
                median_price=(PRICE_COL, "median"),
                min_price=(PRICE_COL, "min"),
                max_price=(PRICE_COL, "max"),
                rule_consistent_share=("rule_consistent_250", "mean"),
            )
            .reset_index()
            .sort_values(["treated", "group_label"], kind="stable")
            .reset_index(drop=True)
        )

    def summarize_outcome_groups(outcome_df):
        """Summarize one outcome's delta by comparison group.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.

        Returns
        -------
        pandas.DataFrame
            Group-level delta and level summaries.

        """
        return (
            outcome_df.groupby(["group_label", "treated"], observed=True)
            .agg(
                outcome=("outcome", "first"),
                transform=("transform", "first"),
                n_obs=("symbol", "size"),
                mean_delta=("delta_y", "mean"),
                median_delta=("delta_y", "median"),
                std_delta=("delta_y", "std"),
                mean_y_pre=("y_pre", "mean"),
                mean_y_post=("y_post", "mean"),
                mean_price=(PRICE_COL, "mean"),
                median_price=(PRICE_COL, "median"),
                rule_consistent_share=("rule_consistent_250", "mean"),
            )
            .reset_index()
            .sort_values(["treated", "group_label"], kind="stable")
            .reset_index(drop=True)
        )

    def fit_treated_regression(outcome_df):
        """Estimate the difference in mean changes between the two groups with OLS.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.

        Returns
        -------
        tuple[RegressionResultsWrapper, pandas.DataFrame]
            OLS fit and group summary table.

        """
        group_counts = outcome_df["treated"].value_counts(dropna=False)
        if len(group_counts) < MIN_GROUPS or 0 not in group_counts or 1 not in group_counts:
            message = "Both comparison groups are required for regression."
            raise ValueError(message)

        design_df = pd.DataFrame(
            {
                "const": 1.0,
                "treated": outcome_df["treated"].to_numpy(dtype=float),
            }
        )
        fit = sm.OLS(
            outcome_df["delta_y"].to_numpy(dtype=float),
            design_df,
        ).fit(cov_type="HC1")
        group_summary_df = summarize_outcome_groups(outcome_df)
        return fit, group_summary_df

    def summarize_regression(outcome_df, fit, group_summary_df):
        """Extract one flat regression summary row and one JSON payload.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.
        fit
            Fitted OLS result.
        group_summary_df
            Group-level summary for the outcome.

        Returns
        -------
        tuple[dict[str, object], dict[str, object]]
            Flat summary row and richer JSON payload.

        """
        control_row = group_summary_df.loc[group_summary_df["treated"] == 0].iloc[0]
        treated_row = group_summary_df.loc[group_summary_df["treated"] == 1].iloc[0]
        coef_treated = float(fit.params["treated"])
        se_treated = float(fit.bse["treated"])
        t_treated = float(fit.tvalues["treated"])
        p_treated = float(fit.pvalues["treated"])
        ci_low, ci_high = fit.conf_int().loc["treated"].tolist()

        flat_summary = {
            "status": "ok",
            "outcome": str(outcome_df["outcome"].iloc[0]),
            "transform": str(outcome_df["transform"].iloc[0]),
            "coef_treated": coef_treated,
            "se_treated": se_treated,
            "t_treated": t_treated,
            "p_value": p_treated,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_obs": len(outcome_df),
            "n_treated": int(treated_row["n_obs"]),
            "n_control": int(control_row["n_obs"]),
            "r2": float(fit.rsquared),
            "mean_delta_control": float(control_row["mean_delta"]),
            "mean_delta_treated": float(treated_row["mean_delta"]),
            "rule_consistent_share_control": float(control_row["rule_consistent_share"]),
            "rule_consistent_share_treated": float(treated_row["rule_consistent_share"]),
        }
        json_summary = {
            **flat_summary,
            "coef_const": float(fit.params["const"]),
            "se_const": float(fit.bse["const"]),
            "group_summary_rows": group_summary_df.to_dict(orient="records"),
        }
        return flat_summary, json_summary

    def make_boxplot(outcome_df, *, y_stem, outpath):
        """Create a two-group boxplot with jittered raw points.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.
        y_stem
            Outcome stem.
        outpath
            Output PNG path.

        Returns
        -------
        pathlib.Path
            Saved plot path.

        """
        figure, axis = plt.subplots(figsize=(8, 5))
        ordered_groups = ["stay100", "to40"]
        display_labels = {"stay100": "100のまま", "to40": "40へ移行"}
        series_list = [
            outcome_df.loc[outcome_df["group_label"] == group_label, "delta_y"].to_numpy(
                dtype=float
            )
            for group_label in ordered_groups
        ]
        boxplot = axis.boxplot(
            series_list,
            tick_labels=[display_labels[group_label] for group_label in ordered_groups],
            patch_artist=True,
            widths=0.55,
        )
        box_colors = {"stay100": "#93c5fd", "to40": "#fca5a5"}
        for patch, group_label in zip(boxplot["boxes"], ordered_groups, strict=True):
            patch.set_facecolor(box_colors[group_label])
            patch.set_alpha(0.85)

        jitter_rng = np.random.default_rng(20260408)
        for index, group_label in enumerate(ordered_groups, start=1):
            group_values = outcome_df.loc[
                outcome_df["group_label"] == group_label,
                "delta_y",
            ].to_numpy(dtype=float)
            if len(group_values) == 0:
                continue
            x_positions = index + jitter_rng.uniform(-0.08, 0.08, size=len(group_values))
            axis.scatter(
                x_positions,
                group_values,
                color="#111827",
                alpha=0.75,
                s=22,
                linewidths=0,
            )

        axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_title(f"round lot 変更後の2群比較: {y_stem}")
        axis.set_xlabel("事後 round lot の群")
        axis.set_ylabel("アウトカムの変化")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(outpath, dpi=200)
        plt.close(figure)
        return outpath

    def make_coefficient_plot(summary_df, *, outpath):
        """Create a coefficient plot across all successful outcomes.

        Parameters
        ----------
        summary_df
            Aggregate summary table with one row per outcome.
        outpath
            Output PNG path.

        Returns
        -------
        pathlib.Path
            Saved plot path.

        """
        ok_df = summary_df.loc[summary_df["status"] == "ok"].copy()
        if ok_df.empty:
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.text(0.5, 0.5, "No successful regressions", ha="center", va="center")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(outpath, dpi=200)
            plt.close(figure)
            return outpath

        ok_df = ok_df.sort_values("coef_treated", kind="stable").reset_index(drop=True)
        y_positions = np.arange(len(ok_df))
        lower_errors = ok_df["coef_treated"] - ok_df["ci_low"]
        upper_errors = ok_df["ci_high"] - ok_df["coef_treated"]

        figure, axis = plt.subplots(figsize=(9, max(4.5, 0.65 * len(ok_df) + 1.5)))
        axis.errorbar(
            ok_df["coef_treated"],
            y_positions,
            xerr=[lower_errors, upper_errors],
            fmt="o",
            color="#1d4ed8",
            ecolor="#60a5fa",
            elinewidth=1.5,
            capsize=3,
        )
        axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(ok_df["outcome"].tolist())
        axis.set_xlabel("平均変化の差（40へ移行した群 - 100のままの群）")
        axis.set_title("round lot 変更比較の係数プロット")
        axis.grid(alpha=0.2, axis="x")
        figure.tight_layout()
        figure.savefig(outpath, dpi=200)
        plt.close(figure)
        return outpath

    def json_default(value):
        """Convert analysis outputs into JSON-safe values.

        Parameters
        ----------
        value
            Object passed to `json.dump`.

        Returns
        -------
        object
            JSON-safe representation.

        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, datetime):
            return value.isoformat()
        message = f"Object of type {type(value)!r} is not JSON serializable."
        raise TypeError(message)

    def write_json(obj, path):
        """Write one JSON artifact with stable formatting.

        Parameters
        ----------
        obj
            JSON-serializable object.
        path
            Output path.

        Returns
        -------
        pathlib.Path
            Written output path.

        """
        with path.open("w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, default=json_default)
        return path

    return (
        CONTROL_POST_ROUND_LOT,
        DATE_POST,
        DATE_PRE,
        POST_RL_COL,
        PRE_RL_COL,
        PRICE_COL,
        TREATED_POST_ROUND_LOT,
        build_base_sample,
        build_outcome_sample,
        fit_treated_regression,
        infer_available_y_stems,
        make_boxplot,
        make_coefficient_plot,
        summarize_base_groups,
        summarize_regression,
        transform_for_outcome,
        write_json,
    )


@app.cell
def _(
    POST_RL_COL,
    PRE_RL_COL,
    PRICE_COL,
    allowed_post_round_lots,
    build_base_sample,
    infer_available_y_stems,
    pd,
    pre_round_lot_required,
    resolved_input_csv_path,
):
    raw_df = pd.read_csv(resolved_input_csv_path)
    available_y_stems = infer_available_y_stems(raw_df.columns)
    base_sample_df = build_base_sample(
        raw_df,
        pre_round_lot_required=pre_round_lot_required,
        allowed_post_round_lots=allowed_post_round_lots,
    )
    transition_scope_df = raw_df.loc[
        raw_df[PRICE_COL].notna()
        & raw_df[PRE_RL_COL].eq(pre_round_lot_required)
        & raw_df[POST_RL_COL].notna(),
        [PRICE_COL, PRE_RL_COL, POST_RL_COL],
    ].copy()
    excluded_post_round_lot_counts = (
        transition_scope_df.loc[
            ~transition_scope_df[POST_RL_COL].isin(list(allowed_post_round_lots))
        ]
        .groupby(POST_RL_COL, observed=True)
        .size()
        .to_dict()
    )
    return available_y_stems, base_sample_df, excluded_post_round_lot_counts, raw_df


@app.cell
def _(base_sample_df, summarize_base_groups):
    base_group_summary_df = summarize_base_groups(base_sample_df)
    return (base_group_summary_df,)


@app.cell
def _(  # noqa: PLR0915
    CONTROL_POST_ROUND_LOT,
    DATE_POST,
    DATE_PRE,
    POST_RL_COL,
    PRE_RL_COL,
    PRICE_COL,
    Path,
    TREATED_POST_ROUND_LOT,
    UTC,
    allowed_post_round_lots,
    available_y_stems,
    base_sample_df,
    build_outcome_sample,
    datetime,
    export_analysis_sample,
    export_group_summary,
    fit_treated_regression,
    make_boxplot,
    make_coefficient_plot,
    panel_csv_stem,
    pd,
    pre_round_lot_required,
    raw_df,
    resolved_input_csv_path,
    resolved_output_dir,
    summarize_regression,
    transform_for_outcome,
    write_json,
    excluded_post_round_lot_counts,
):
    summary_rows = []
    artifact_rows = []
    preview_plot_path = None

    for y_stem in available_y_stems:
        summary_path = resolved_output_dir / f"regression__{y_stem}.json"
        try:
            outcome_df = build_outcome_sample(base_sample_df, y_stem=y_stem)
            fit, group_summary_df = fit_treated_regression(outcome_df)
            flat_summary, json_summary = summarize_regression(
                outcome_df,
                fit=fit,
                group_summary_df=group_summary_df,
            )

            analysis_sample_path = None
            if export_analysis_sample:
                analysis_sample_path = resolved_output_dir / f"analysis_sample__{y_stem}.csv"
                outcome_df.to_csv(analysis_sample_path, index=False)
                artifact_rows.append(
                    {
                        "outcome": y_stem,
                        "artifact_type": "analysis_sample_csv",
                        "path": str(analysis_sample_path),
                    }
                )

            group_summary_path = None
            if export_group_summary:
                group_summary_path = resolved_output_dir / f"group_summary__{y_stem}.csv"
                group_summary_df.to_csv(group_summary_path, index=False)
                artifact_rows.append(
                    {
                        "outcome": y_stem,
                        "artifact_type": "group_summary_csv",
                        "path": str(group_summary_path),
                    }
                )

            boxplot_path = resolved_output_dir / f"boxplot__{y_stem}.png"
            make_boxplot(outcome_df, y_stem=y_stem, outpath=boxplot_path)
            if preview_plot_path is None:
                preview_plot_path = boxplot_path
            artifact_rows.append(
                {
                    "outcome": y_stem,
                    "artifact_type": "boxplot_png",
                    "path": str(boxplot_path),
                }
            )

            json_summary["analysis_sample_csv"] = (
                None if analysis_sample_path is None else str(analysis_sample_path)
            )
            json_summary["group_summary_csv"] = (
                None if group_summary_path is None else str(group_summary_path)
            )
            json_summary["boxplot_png"] = str(boxplot_path)
            json_summary["regression_json"] = str(summary_path)
            write_json(json_summary, summary_path)
            artifact_rows.append(
                {
                    "outcome": y_stem,
                    "artifact_type": "regression_json",
                    "path": str(summary_path),
                }
            )

            flat_summary["analysis_sample_csv"] = (
                None if analysis_sample_path is None else str(analysis_sample_path)
            )
            flat_summary["group_summary_csv"] = (
                None if group_summary_path is None else str(group_summary_path)
            )
            flat_summary["boxplot_png"] = str(boxplot_path)
            flat_summary["regression_json"] = str(summary_path)
            summary_rows.append(flat_summary)
        except Exception as exc:  # noqa: BLE001
            error_summary = {
                "status": "error",
                "outcome": y_stem,
                "transform": transform_for_outcome(y_stem),
                "coef_treated": None,
                "se_treated": None,
                "t_treated": None,
                "p_value": None,
                "ci_low": None,
                "ci_high": None,
                "n_obs": 0,
                "n_treated": 0,
                "n_control": 0,
                "r2": None,
                "mean_delta_control": None,
                "mean_delta_treated": None,
                "rule_consistent_share_control": None,
                "rule_consistent_share_treated": None,
                "analysis_sample_csv": None,
                "group_summary_csv": None,
                "boxplot_png": None,
                "regression_json": str(summary_path),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            write_json(error_summary, summary_path)
            artifact_rows.append(
                {
                    "outcome": y_stem,
                    "artifact_type": "regression_json",
                    "path": str(summary_path),
                }
            )
            summary_rows.append(error_summary)

    analysis_results_df = pd.DataFrame(summary_rows)
    summary_all_csv_path = resolved_output_dir / "summary_all.csv"
    analysis_results_df.to_csv(summary_all_csv_path, index=False)
    artifact_rows.append(
        {
            "outcome": "ALL",
            "artifact_type": "summary_all_csv",
            "path": str(summary_all_csv_path),
        }
    )

    summary_all_json_path = resolved_output_dir / "summary_all.json"
    write_json({"rows": summary_rows}, summary_all_json_path)
    artifact_rows.append(
        {
            "outcome": "ALL",
            "artifact_type": "summary_all_json",
            "path": str(summary_all_json_path),
        }
    )

    coef_plot_path = resolved_output_dir / "coef_plot.png"
    make_coefficient_plot(analysis_results_df, outpath=coef_plot_path)
    artifact_rows.append(
        {
            "outcome": "ALL",
            "artifact_type": "coef_plot_png",
            "path": str(coef_plot_path),
        }
    )

    artifact_manifest_df = (
        pd.DataFrame(artifact_rows)
        .sort_values(["outcome", "artifact_type", "path"], kind="stable")
        .reset_index(drop=True)
    )
    artifact_manifest_df["exists"] = artifact_manifest_df["path"].map(
        lambda path: Path(path).exists()
    )

    run_manifest = {
        "created_at_utc": datetime.now(UTC),
        "input_csv_path": str(resolved_input_csv_path),
        "panel_csv_stem": panel_csv_stem,
        "output_dir": str(resolved_output_dir),
        "date_pre": DATE_PRE,
        "date_post": DATE_POST,
        "price_col": PRICE_COL,
        "pre_round_lot_col": PRE_RL_COL,
        "post_round_lot_col": POST_RL_COL,
        "pre_round_lot_required": pre_round_lot_required,
        "allowed_post_round_lots": list(allowed_post_round_lots),
        "control_post_round_lot": CONTROL_POST_ROUND_LOT,
        "treated_post_round_lot": TREATED_POST_ROUND_LOT,
        "available_y_stems": list(available_y_stems),
        "n_raw_rows": len(raw_df),
        "n_base_sample": len(base_sample_df),
        "n_successful_outcomes": int((analysis_results_df["status"] == "ok").sum()),
        "excluded_post_round_lot_counts": excluded_post_round_lot_counts,
        "artifact_files": artifact_manifest_df.to_dict(orient="records"),
        "analysis_note": "Observed actual group comparison only; not a causal RD design.",
    }
    run_manifest_path = resolved_output_dir / "run_manifest.json"
    write_json(run_manifest, run_manifest_path)
    artifact_manifest_df = pd.concat(
        [
            artifact_manifest_df,
            pd.DataFrame(
                [
                    {
                        "outcome": "ALL",
                        "artifact_type": "run_manifest_json",
                        "path": str(run_manifest_path),
                        "exists": run_manifest_path.exists(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    input_overview_df = pd.DataFrame(
        [
            {
                "input_csv_path": str(resolved_input_csv_path),
                "output_dir": str(resolved_output_dir),
                "n_raw_rows": len(raw_df),
                "n_base_sample": len(base_sample_df),
                "n_outcomes": len(available_y_stems),
            }
        ]
    )
    outcome_list_df = pd.DataFrame({"outcome": available_y_stems})
    return (
        analysis_results_df,
        artifact_manifest_df,
        coef_plot_path,
        input_overview_df,
        outcome_list_df,
        preview_plot_path,
        run_manifest,
    )


@app.cell
def _():
    analysis_note = (
        "この notebook は，各アウトカムの変化を，round lot が 100 のままの銘柄群と "
        "100 から 40 へ移行した銘柄群のあいだで比較する．"
        "これは実際の群どうしの比較であり，因果的な RD 設計ではない．"
    )
    analysis_note  # noqa: B018
    return


@app.cell
def _(input_overview_df):
    input_overview_df  # noqa: B018
    return


@app.cell
def _(outcome_list_df):
    outcome_list_df  # noqa: B018
    return


@app.cell
def _(base_group_summary_df):
    base_group_summary_df  # noqa: B018
    return


@app.cell
def _(analysis_results_df):
    analysis_results_df  # noqa: B018
    return


@app.cell
def _(artifact_manifest_df):
    artifact_manifest_df  # noqa: B018
    return


@app.cell
def _(coef_plot_path, mpimg, plt):
    coef_plot_image = mpimg.imread(coef_plot_path)
    coef_plot_preview, _axis = plt.subplots(figsize=(9, 5))
    _axis.imshow(coef_plot_image)
    _axis.axis("off")
    _axis.set_title(str(coef_plot_path))
    coef_plot_preview.tight_layout()
    coef_plot_preview  # noqa: B018
    return


@app.cell
def _(mpimg, plt, preview_plot_path):
    if preview_plot_path is None:
        preview_plot = None
    else:
        preview_image = mpimg.imread(preview_plot_path)
        preview_plot, _axis = plt.subplots(figsize=(8, 5))
        _axis.imshow(preview_image)
        _axis.axis("off")
        _axis.set_title(str(preview_plot_path))
        preview_plot.tight_layout()
    preview_plot  # noqa: B018
    return


@app.cell
def _(run_manifest):
    run_manifest  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
