import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from taq_llag_analysis.preprocess.daily_taq_paths import master_input_path

    plt.rcParams["axes.unicode_minus"] = False
    return Path, UTC, datetime, json, master_input_path, mpimg, np, pd, plt, sm


@app.cell
def _(Path):
    input_csv_path = Path("../data/derived/trade_quote_modes/my-run/cross_k_summary.csv")
    output_base_dir = Path("../data/derived/cross_k_round_lot_effects")
    trade_exchange = "Z"
    quote_exchange = "Q"
    date_pre = "20251031"
    date_post = "20251103"
    pre_round_lot_required = 100.0
    allowed_post_round_lots = (100.0, 40.0)
    export_analysis_sample = True
    export_group_summary = True
    cross_k_columns = [
        "cross_k_neg_1e3_1e4",
        "cross_k_neg_1e4_1e5",
        "cross_k_neg_1e5_0",
        "cross_k_pos_0_1e5",
        "cross_k_pos_1e5_1e4",
        "cross_k_pos_1e4_1e3",
    ]
    return (
        allowed_post_round_lots,
        cross_k_columns,
        date_post,
        date_pre,
        export_analysis_sample,
        export_group_summary,
        input_csv_path,
        output_base_dir,
        pre_round_lot_required,
        quote_exchange,
        trade_exchange,
    )


@app.cell
def _(Path, input_csv_path, output_base_dir, quote_exchange, trade_exchange):
    resolved_input_csv_path = Path(input_csv_path)
    if not resolved_input_csv_path.exists():
        message = f"cross_k summary not found: {resolved_input_csv_path}"
        raise FileNotFoundError(message)

    input_csv_stem = resolved_input_csv_path.stem
    pair_label = f"{trade_exchange}{quote_exchange}"
    resolved_output_dir = output_base_dir / f"{input_csv_stem}_{pair_label}"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return (
        input_csv_stem,
        pair_label,
        resolved_input_csv_path,
        resolved_output_dir,
    )


@app.cell
def _(
    Path,
    date_post,
    date_pre,
    datetime,
    json,
    master_input_path,
    np,
    pd,
    plt,
    sm,
):
    PRE_RL_COL = f"round_lot_{date_pre}"
    POST_RL_COL = f"round_lot_{date_post}"
    MIN_GROUPS = 2

    def build_cross_k_panel(
        summary_df,
        *,
        trade_exchange,
        quote_exchange,
        date_pre,
        date_post,
        cross_k_columns,
    ):
        """Build a symbol-level wide panel for the selected cross-k outcomes.

        Parameters
        ----------
        summary_df
            Long-form `cross_k_summary.csv` dataframe.
        trade_exchange
            Trade exchange code to keep.
        quote_exchange
            Quote exchange code to keep.
        date_pre
            Pre-reform date in `YYYYMMDD` format.
        date_post
            Post-reform date in `YYYYMMDD` format.
        cross_k_columns
            Outcome column names to pivot into wide form.

        Returns
        -------
        pandas.DataFrame
            Symbol-level panel with one pre/post column pair per outcome.

        """
        required_columns = [
            "symbol",
            "date_yyyymmdd",
            "status",
            "trade_exchange",
            "quote_exchange",
            *cross_k_columns,
        ]
        missing_columns = [
            column_name for column_name in required_columns if column_name not in summary_df.columns
        ]
        if missing_columns:
            message = f"Missing required columns: {missing_columns}"
            raise KeyError(message)

        selected_df = summary_df.copy()
        selected_df["date_yyyymmdd"] = selected_df["date_yyyymmdd"].astype(str)
        selected_df = selected_df.loc[
            (selected_df["status"] == "ok")
            & (selected_df["trade_exchange"] == trade_exchange)
            & (selected_df["quote_exchange"] == quote_exchange)
            & selected_df["date_yyyymmdd"].isin([date_pre, date_post]),
            ["symbol", "date_yyyymmdd", *cross_k_columns],
        ].copy()

        wide_parts = []
        for column_name in cross_k_columns:
            wide_part = selected_df.pivot_table(
                index="symbol",
                columns="date_yyyymmdd",
                values=column_name,
                aggfunc="first",
            )
            wide_part = wide_part.reindex(columns=[date_pre, date_post])
            wide_part = wide_part.rename(
                columns={
                    date_pre: f"{column_name}_{date_pre}",
                    date_post: f"{column_name}_{date_post}",
                },
            )
            wide_parts.append(wide_part)

        wide_df = pd.concat(wide_parts, axis=1)
        wide_df.index.name = "symbol"
        return wide_df.sort_index().reset_index()

    def load_round_lot_panel(*, date_pre, date_post):
        """Load round-lot values from Daily TAQ master files.

        Parameters
        ----------
        date_pre
            Pre-reform date in `YYYYMMDD` format.
        date_post
            Post-reform date in `YYYYMMDD` format.

        Returns
        -------
        pandas.DataFrame
            Symbol-level wide dataframe with pre/post round-lot columns.

        """
        master_frames = []
        for date_yyyymmdd in (date_pre, date_post):
            master_path = master_input_path(date_yyyymmdd)
            if not master_path.exists():
                message = f"Master file not found: {master_path}"
                raise FileNotFoundError(message)

            master_df = pd.read_csv(
                master_path,
                sep="|",
                encoding="latin1",
                compression="gzip",
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

        round_lot_df = pd.concat(master_frames, ignore_index=True)
        wide_round_lot_df = round_lot_df.pivot_table(
            index="symbol",
            columns="date_yyyymmdd",
            values="round_lot",
            aggfunc="first",
        )
        wide_round_lot_df = wide_round_lot_df.reindex(columns=[date_pre, date_post])
        wide_round_lot_df = wide_round_lot_df.rename(
            columns={
                date_pre: PRE_RL_COL,
                date_post: POST_RL_COL,
            },
        )
        wide_round_lot_df.index.name = "symbol"
        return wide_round_lot_df.sort_index().reset_index()

    def build_base_sample(
        panel_df,
        *,
        pre_round_lot_required,
        allowed_post_round_lots,
    ):
        """Construct the common comparison sample before outcome-specific filtering.

        Parameters
        ----------
        panel_df
            Wide symbol-level panel with cross-k outcomes and round-lot columns.
        pre_round_lot_required
            Required pre-reform round lot.
        allowed_post_round_lots
            Post-reform round lots kept in the comparison sample.

        Returns
        -------
        pandas.DataFrame
            Base comparison sample with treatment assignment columns.

        """
        required_columns = ["symbol", PRE_RL_COL, POST_RL_COL]
        missing_columns = [
            column_name for column_name in required_columns if column_name not in panel_df.columns
        ]
        if missing_columns:
            message = f"Missing required columns: {missing_columns}"
            raise KeyError(message)

        sample_df = panel_df.copy()
        sample_df = sample_df.loc[sample_df[PRE_RL_COL] == pre_round_lot_required].copy()
        sample_df = sample_df.loc[sample_df[POST_RL_COL].isin(list(allowed_post_round_lots))].copy()
        sample_df["treated"] = (sample_df[POST_RL_COL] == 40.0).astype(int)
        sample_df["group_label"] = np.where(sample_df["treated"] == 1, "to40", "stay100")
        sample_df["actual_transition"] = (
            sample_df[PRE_RL_COL].astype("Int64").astype(str)
            + "->"
            + sample_df[POST_RL_COL].astype("Int64").astype(str)
        )
        return sample_df.sort_values("symbol", kind="stable").reset_index(drop=True)

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
        pre_y_col = f"{y_stem}_{date_pre}"
        post_y_col = f"{y_stem}_{date_post}"
        missing_columns = [
            column_name
            for column_name in (pre_y_col, post_y_col)
            if column_name not in base_sample_df.columns
        ]
        if missing_columns:
            message = f"Missing outcome columns: {missing_columns}"
            raise KeyError(message)

        outcome_df = base_sample_df.copy()
        outcome_df["y_pre"] = pd.to_numeric(outcome_df[pre_y_col], errors="coerce")
        outcome_df["y_post"] = pd.to_numeric(outcome_df[post_y_col], errors="coerce")
        outcome_df["delta_y"] = np.log1p(outcome_df["y_post"]) - np.log1p(outcome_df["y_pre"])
        outcome_df = outcome_df.loc[
            outcome_df["y_pre"].notna()
            & outcome_df["y_post"].notna()
            & outcome_df["delta_y"].notna()
        ].copy()
        outcome_df["outcome"] = y_stem
        outcome_df["transform"] = "log1p"
        return outcome_df

    def summarize_base_groups(base_sample_df):
        """Summarize the common comparison sample by post-reform round-lot group.

        Parameters
        ----------
        base_sample_df
            Common comparison sample.

        Returns
        -------
        pandas.DataFrame
            Group counts and transition labels.

        """
        return (
            base_sample_df.groupby(["group_label", "treated"], observed=True)
            .agg(
                actual_transition=("actual_transition", "first"),
                n_obs=("symbol", "size"),
                n_symbols=("symbol", "nunique"),
            )
            .reset_index()
            .sort_values(["treated", "group_label"], kind="stable")
            .reset_index(drop=True)
        )

    def summarize_outcome_groups(outcome_df):
        """Summarize one outcome's change by comparison group.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.

        Returns
        -------
        pandas.DataFrame
            Group-level summaries for the outcome.

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
            )
            .reset_index()
            .sort_values(["treated", "group_label"], kind="stable")
            .reset_index(drop=True)
        )

    def fit_treated_regression(outcome_df):
        """Estimate the difference in mean outcome changes with OLS.

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
        display_labels = {"stay100": "stay100", "to40": "to40"}
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

        jitter_rng = np.random.default_rng(20260418)
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
        axis.set_title(f"Round-lot comparison: {y_stem}")
        axis.set_xlabel("Post-reform round-lot group")
        axis.set_ylabel("log1p(post) - log1p(pre)")
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
        axis.set_xlabel("Difference in mean log1p change (to40 - stay100)")
        axis.set_title("Cross-K round-lot comparison")
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
        POST_RL_COL,
        PRE_RL_COL,
        build_base_sample,
        build_cross_k_panel,
        build_outcome_sample,
        fit_treated_regression,
        load_round_lot_panel,
        make_boxplot,
        make_coefficient_plot,
        summarize_base_groups,
        summarize_regression,
        write_json,
    )


@app.cell
def _(
    POST_RL_COL,
    PRE_RL_COL,
    allowed_post_round_lots,
    build_base_sample,
    build_cross_k_panel,
    cross_k_columns,
    date_post,
    date_pre,
    load_round_lot_panel,
    pd,
    pre_round_lot_required,
    quote_exchange,
    resolved_input_csv_path,
    trade_exchange,
):
    raw_summary_df = pd.read_csv(resolved_input_csv_path)
    raw_summary_df["date_yyyymmdd"] = raw_summary_df["date_yyyymmdd"].astype(str)

    pair_ok_df = raw_summary_df.loc[
        (raw_summary_df["status"] == "ok")
        & (raw_summary_df["trade_exchange"] == trade_exchange)
        & (raw_summary_df["quote_exchange"] == quote_exchange)
        & raw_summary_df["date_yyyymmdd"].isin([date_pre, date_post]),
    ].copy()

    wide_cross_k_df = build_cross_k_panel(
        raw_summary_df,
        trade_exchange=trade_exchange,
        quote_exchange=quote_exchange,
        date_pre=date_pre,
        date_post=date_post,
        cross_k_columns=cross_k_columns,
    )
    round_lot_panel_df = load_round_lot_panel(date_pre=date_pre, date_post=date_post)
    panel_df = wide_cross_k_df.merge(round_lot_panel_df, on="symbol", how="left")

    available_y_stems = list(cross_k_columns)
    base_sample_df = build_base_sample(
        panel_df,
        pre_round_lot_required=pre_round_lot_required,
        allowed_post_round_lots=allowed_post_round_lots,
    )

    transition_scope_df = panel_df.loc[
        panel_df[PRE_RL_COL].eq(pre_round_lot_required) & panel_df[POST_RL_COL].notna(),
        ["symbol", PRE_RL_COL, POST_RL_COL],
    ].copy()
    excluded_post_round_lot_counts = (
        transition_scope_df.loc[
            ~transition_scope_df[POST_RL_COL].isin(list(allowed_post_round_lots))
        ]
        .groupby(POST_RL_COL, observed=True)
        .size()
        .to_dict()
    )
    return (
        available_y_stems,
        base_sample_df,
        excluded_post_round_lot_counts,
        pair_ok_df,
        panel_df,
        raw_summary_df,
        round_lot_panel_df,
    )


@app.cell
def _(base_sample_df, summarize_base_groups):
    base_group_summary_df = summarize_base_groups(base_sample_df)
    return (base_group_summary_df,)


@app.cell
def _(
    POST_RL_COL,
    PRE_RL_COL,
    Path,
    UTC,
    allowed_post_round_lots,
    available_y_stems,
    base_sample_df,
    build_outcome_sample,
    date_post,
    date_pre,
    datetime,
    excluded_post_round_lot_counts,
    export_analysis_sample,
    export_group_summary,
    fit_treated_regression,
    input_csv_stem,
    make_boxplot,
    make_coefficient_plot,
    pair_label,
    pair_ok_df,
    panel_df,
    pd,
    pre_round_lot_required,
    quote_exchange,
    raw_summary_df,
    resolved_input_csv_path,
    resolved_output_dir,
    round_lot_panel_df,
    summarize_regression,
    trade_exchange,
    write_json,
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
                "transform": "log1p",
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
        "input_csv_stem": input_csv_stem,
        "output_dir": str(resolved_output_dir),
        "pair_label": pair_label,
        "trade_exchange": trade_exchange,
        "quote_exchange": quote_exchange,
        "date_pre": date_pre,
        "date_post": date_post,
        "pre_round_lot_col": PRE_RL_COL,
        "post_round_lot_col": POST_RL_COL,
        "pre_round_lot_required": pre_round_lot_required,
        "allowed_post_round_lots": list(allowed_post_round_lots),
        "available_y_stems": list(available_y_stems),
        "n_raw_rows": len(raw_summary_df),
        "n_pair_ok_rows": len(pair_ok_df),
        "n_panel_symbols": int(panel_df["symbol"].nunique()),
        "n_round_lot_symbols": int(round_lot_panel_df["symbol"].nunique()),
        "n_base_sample": len(base_sample_df),
        "n_successful_outcomes": int((analysis_results_df["status"] == "ok").sum()),
        "excluded_post_round_lot_counts": excluded_post_round_lot_counts,
        "artifact_files": artifact_manifest_df.to_dict(orient="records"),
        "analysis_note": ("Observed actual group comparison only; not a causal RD design."),
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
                "pair_label": pair_label,
                "n_raw_rows": len(raw_summary_df),
                "n_pair_ok_rows": len(pair_ok_df),
                "n_panel_symbols": int(panel_df["symbol"].nunique()),
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
    )


@app.cell
def _():
    analysis_note = (
        "この notebook は，各 cross K 指標の変化を，round lot が 100 のままの銘柄群と "
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


if __name__ == "__main__":
    app.run()
