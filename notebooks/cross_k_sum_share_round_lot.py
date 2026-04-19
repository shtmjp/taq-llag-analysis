import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    import marimo as mo
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
    return (
        Path,
        UTC,
        datetime,
        json,
        master_input_path,
        mo,
        mpimg,
        np,
        pd,
        plt,
        sm,
    )


@app.cell
def _(mo):
    analysis_note = "Observed actual round-lot transition comparison only; not a causal RD design."
    overview_section = mo.md(
        f"""
        ## Overview

        This notebook aggregates the negative-side and positive-side cross-K
        windows into totals and within-side shares, then compares observed
        round-lot transition groups around the November 3, 2025 reform.

        `{analysis_note}`
        """
    )
    overview_section
    return (analysis_note,)


@app.cell
def _(Path, pd):
    input_csv_path = Path("../data/derived/trade_quote_modes/my-run/cross_k_summary.csv")
    output_base_dir = Path("output/cross_k_sum_share_round_lot_effects")
    trade_exchange = "N"
    quote_exchange = "P"
    date_pre = "20251031"
    date_post = "20251103"
    pre_round_lot_required = 100.0
    allowed_post_round_lots = (100.0, 40.0, 10.0)
    group_labels = ("stay100", "to40", "to10")
    export_analysis_sample = True
    export_group_summary = True
    component_groups = {
        "negative": (
            {
                "distance_band": "far_from_zero",
                "window_label": "[-1e-3, -1e-4)",
                "raw_column": "cross_k_neg_1e3_1e4",
                "share_outcome": "negative_far_from_zero_share",
            },
            {
                "distance_band": "mid_range",
                "window_label": "[-1e-4, -1e-5)",
                "raw_column": "cross_k_neg_1e4_1e5",
                "share_outcome": "negative_mid_range_share",
            },
            {
                "distance_band": "near_zero",
                "window_label": "[-1e-5, 0)",
                "raw_column": "cross_k_neg_1e5_0",
                "share_outcome": "negative_near_zero_share",
            },
        ),
        "positive": (
            {
                "distance_band": "near_zero",
                "window_label": "[0, 1e-5)",
                "raw_column": "cross_k_pos_0_1e5",
                "share_outcome": "positive_near_zero_share",
            },
            {
                "distance_band": "mid_range",
                "window_label": "[1e-5, 1e-4)",
                "raw_column": "cross_k_pos_1e5_1e4",
                "share_outcome": "positive_mid_range_share",
            },
            {
                "distance_band": "far_from_zero",
                "window_label": "[1e-4, 1e-3)",
                "raw_column": "cross_k_pos_1e4_1e3",
                "share_outcome": "positive_far_from_zero_share",
            },
        ),
    }
    sum_outcomes = {
        "negative": "negative_total",
        "positive": "positive_total",
    }
    raw_cross_k_columns = [
        component["raw_column"]
        for side_components in component_groups.values()
        for component in side_components
    ]
    direct_outcome_specs = [
        {
            "raw_column": "closest_mode_to_zero_sec",
            "outcome": "mode_position_sec",
            "display_name": "Mode position (sec)",
            "side": "overall",
            "family": "mode",
            "transform": "signed",
            "window_label": "Signed nearest-to-zero mode location",
        }
    ]
    raw_panel_columns = [
        *raw_cross_k_columns,
        *[spec["raw_column"] for spec in direct_outcome_specs],
    ]
    outcome_specs = [
        {
            "outcome": "negative_total",
            "display_name": "Negative total",
            "side": "negative",
            "family": "sum",
            "transform": "log1p",
        },
        {
            "outcome": "positive_total",
            "display_name": "Positive total",
            "side": "positive",
            "family": "sum",
            "transform": "log1p",
        },
        {
            "outcome": "negative_near_zero_share",
            "display_name": "Negative near-zero share",
            "side": "negative",
            "family": "share",
            "transform": "signed",
        },
        {
            "outcome": "negative_mid_range_share",
            "display_name": "Negative mid-range share",
            "side": "negative",
            "family": "share",
            "transform": "signed",
        },
        {
            "outcome": "negative_far_from_zero_share",
            "display_name": "Negative far-from-zero share",
            "side": "negative",
            "family": "share",
            "transform": "signed",
        },
        {
            "outcome": "positive_near_zero_share",
            "display_name": "Positive near-zero share",
            "side": "positive",
            "family": "share",
            "transform": "signed",
        },
        {
            "outcome": "positive_mid_range_share",
            "display_name": "Positive mid-range share",
            "side": "positive",
            "family": "share",
            "transform": "signed",
        },
        {
            "outcome": "positive_far_from_zero_share",
            "display_name": "Positive far-from-zero share",
            "side": "positive",
            "family": "share",
            "transform": "signed",
        },
        *[
            {
                "outcome": spec["outcome"],
                "display_name": spec["display_name"],
                "side": spec["side"],
                "family": spec["family"],
                "transform": spec["transform"],
            }
            for spec in direct_outcome_specs
        ],
    ]
    outcome_specs_df = pd.DataFrame(outcome_specs)
    component_definition_rows = []
    for side, components in component_groups.items():
        for component in components:
            component_definition_rows.append(
                {
                    "construction": "aggregated",
                    "side": side,
                    "distance_band": component["distance_band"],
                    "window_label": component["window_label"],
                    "raw_column": component["raw_column"],
                    "sum_outcome": sum_outcomes[side],
                    "share_outcome": component["share_outcome"],
                    "direct_outcome": None,
                }
            )
    for spec in direct_outcome_specs:
        component_definition_rows.append(
            {
                "construction": "direct",
                "side": spec["side"],
                "distance_band": "closest_to_zero",
                "window_label": spec["window_label"],
                "raw_column": spec["raw_column"],
                "sum_outcome": None,
                "share_outcome": None,
                "direct_outcome": spec["outcome"],
            }
        )
    component_definition_df = pd.DataFrame(component_definition_rows)
    coefficient_plot_specs = [
        {
            "plot_key": "negative_total",
            "outcomes": ("negative_total",),
            "title": "Negative total coefficients",
            "xlabel": "Difference in mean log1p change relative to stay100",
            "filename": "coef_plot__negative_total.png",
        },
        {
            "plot_key": "positive_total",
            "outcomes": ("positive_total",),
            "title": "Positive total coefficients",
            "xlabel": "Difference in mean log1p change relative to stay100",
            "filename": "coef_plot__positive_total.png",
        },
        {
            "plot_key": "negative_shares",
            "outcomes": (
                "negative_near_zero_share",
                "negative_mid_range_share",
                "negative_far_from_zero_share",
            ),
            "title": "Negative share coefficients",
            "xlabel": "Difference in mean share change relative to stay100",
            "filename": "coef_plot__negative_shares.png",
        },
        {
            "plot_key": "positive_shares",
            "outcomes": (
                "positive_near_zero_share",
                "positive_mid_range_share",
                "positive_far_from_zero_share",
            ),
            "title": "Positive share coefficients",
            "xlabel": "Difference in mean share change relative to stay100",
            "filename": "coef_plot__positive_shares.png",
        },
        {
            "plot_key": "mode_position",
            "outcomes": ("mode_position_sec",),
            "title": "Mode position coefficients",
            "xlabel": "Difference in mean mode-position change relative to stay100",
            "filename": "coef_plot__mode_position.png",
        },
    ]
    return (
        allowed_post_round_lots,
        coefficient_plot_specs,
        component_definition_df,
        component_groups,
        date_post,
        date_pre,
        direct_outcome_specs,
        export_analysis_sample,
        export_group_summary,
        group_labels,
        input_csv_path,
        outcome_specs,
        outcome_specs_df,
        output_base_dir,
        pre_round_lot_required,
        quote_exchange,
        raw_panel_columns,
        sum_outcomes,
        trade_exchange,
    )


@app.cell
def _(
    allowed_post_round_lots,
    date_post,
    date_pre,
    input_csv_path,
    mo,
    output_base_dir,
    pre_round_lot_required,
    quote_exchange,
    trade_exchange,
):
    inputs_section = mo.md(
        f"""
        ## Inputs

        This section records the source CSV, exchange-pair filter, analysis dates,
        and round-lot rules used to build the comparison sample.

        - `input_csv_path`: `{input_csv_path}`
        - `output_base_dir`: `{output_base_dir}`
        - `trade_exchange`: `{trade_exchange}`
        - `quote_exchange`: `{quote_exchange}`
        - `date_pre`: `{date_pre}`
        - `date_post`: `{date_post}`
        - `pre_round_lot_required`: `{pre_round_lot_required}`
        - `allowed_post_round_lots`: `{allowed_post_round_lots}`
        """
    )
    inputs_section
    return


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
    REQUIRED_GROUPS = ("stay100", "to40", "to10")

    def build_cross_k_panel(
        summary_df,
        *,
        trade_exchange,
        quote_exchange,
        date_pre,
        date_post,
        raw_columns,
    ):
        """Build a symbol-level wide panel for selected raw outcome columns.

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
        raw_columns
            Raw outcome column names to pivot into wide form.

        Returns
        -------
        pandas.DataFrame
            Symbol-level panel with one pre/post column pair per raw component.

        """
        selected_df = summary_df.copy()
        selected_df["date_yyyymmdd"] = selected_df["date_yyyymmdd"].astype(str)
        selected_df = selected_df.loc[
            (selected_df["status"] == "ok")
            & (selected_df["trade_exchange"] == trade_exchange)
            & (selected_df["quote_exchange"] == quote_exchange)
            & selected_df["date_yyyymmdd"].isin([date_pre, date_post]),
            ["symbol", "date_yyyymmdd", *raw_columns],
        ].copy()

        wide_parts = []
        for column_name in raw_columns:
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

    def derive_aggregated_outcomes(
        raw_panel_df,
        *,
        component_groups,
        date_pre,
        date_post,
        direct_outcome_specs,
        sum_outcomes,
    ):
        """Construct derived outcomes from the raw cross-k and mode columns.

        Parameters
        ----------
        raw_panel_df
            Wide symbol-level dataframe with raw component columns.
        component_groups
            Mapping from side name to component metadata.
        date_pre
            Pre-reform date in `YYYYMMDD` format.
        date_post
            Post-reform date in `YYYYMMDD` format.
        direct_outcome_specs
            Metadata for direct outcomes copied from raw columns.
        sum_outcomes
            Mapping from side name to total-outcome name.

        Returns
        -------
        pandas.DataFrame
            Symbol-level dataframe with derived pre/post outcomes.

        """
        aggregated_df = raw_panel_df.loc[:, ["symbol"]].copy()
        for side, components in component_groups.items():
            for date_yyyymmdd in (date_pre, date_post):
                component_columns = [
                    f"{component['raw_column']}_{date_yyyymmdd}" for component in components
                ]
                total_column = f"{sum_outcomes[side]}_{date_yyyymmdd}"
                aggregated_df[total_column] = raw_panel_df[component_columns].sum(
                    axis=1,
                    min_count=len(component_columns),
                )
                total_series = aggregated_df[total_column]
                nonzero_total = total_series.where(total_series.ne(0.0))
                for component in components:
                    numerator_column = f"{component['raw_column']}_{date_yyyymmdd}"
                    share_column = f"{component['share_outcome']}_{date_yyyymmdd}"
                    aggregated_df[share_column] = raw_panel_df[numerator_column].div(nonzero_total)
        for spec in direct_outcome_specs:
            for date_yyyymmdd in (date_pre, date_post):
                aggregated_df[f"{spec['outcome']}_{date_yyyymmdd}"] = raw_panel_df[
                    f"{spec['raw_column']}_{date_yyyymmdd}"
                ]
        return aggregated_df

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
            Wide symbol-level panel with aggregated outcomes and round-lot columns.
        pre_round_lot_required
            Required pre-reform round lot.
        allowed_post_round_lots
            Post-reform round lots kept in the comparison sample.

        Returns
        -------
        pandas.DataFrame
            Base comparison sample with group assignment columns.

        """
        sample_df = panel_df.copy()
        sample_df = sample_df.loc[sample_df[PRE_RL_COL] == pre_round_lot_required].copy()
        sample_df = sample_df.loc[sample_df[POST_RL_COL].isin(list(allowed_post_round_lots))].copy()
        sample_df["treated_to40"] = (sample_df[POST_RL_COL] == 40.0).astype(int)
        sample_df["treated_to10"] = (sample_df[POST_RL_COL] == 10.0).astype(int)
        sample_df["group_label"] = np.select(
            [
                sample_df["treated_to40"] == 1,
                sample_df["treated_to10"] == 1,
            ],
            ["to40", "to10"],
            default="stay100",
        )
        sample_df["group_order"] = sample_df["group_label"].map(
            {"stay100": 0, "to40": 1, "to10": 2}
        )
        sample_df["actual_transition"] = (
            sample_df[PRE_RL_COL].astype("Int64").astype(str)
            + "->"
            + sample_df[POST_RL_COL].astype("Int64").astype(str)
        )
        return sample_df.sort_values(["group_order", "symbol"], kind="stable").reset_index(
            drop=True
        )

    def build_outcome_sample(base_sample_df, *, outcome_name, outcome_specs_by_name):
        """Attach one outcome's pre, post, and delta columns to the base sample.

        Parameters
        ----------
        base_sample_df
            Common comparison sample.
        outcome_name
            Aggregated outcome name.
        outcome_specs_by_name
            Outcome metadata keyed by outcome name.

        Returns
        -------
        pandas.DataFrame
            Outcome-specific comparison sample.

        """
        outcome_spec = outcome_specs_by_name[outcome_name]
        pre_y_col = f"{outcome_name}_{date_pre}"
        post_y_col = f"{outcome_name}_{date_post}"

        outcome_df = base_sample_df.copy()
        outcome_df["y_pre"] = pd.to_numeric(outcome_df[pre_y_col], errors="coerce")
        outcome_df["y_post"] = pd.to_numeric(outcome_df[post_y_col], errors="coerce")
        transform = str(outcome_spec["transform"])
        if transform == "log1p":
            outcome_df["delta_y"] = np.log1p(outcome_df["y_post"]) - np.log1p(outcome_df["y_pre"])
        else:
            outcome_df["delta_y"] = outcome_df["y_post"] - outcome_df["y_pre"]

        outcome_df = outcome_df.loc[
            outcome_df["y_pre"].notna()
            & outcome_df["y_post"].notna()
            & outcome_df["delta_y"].notna()
        ].copy()
        outcome_df["outcome"] = outcome_name
        outcome_df["display_name"] = str(outcome_spec["display_name"])
        outcome_df["side"] = str(outcome_spec["side"])
        outcome_df["family"] = str(outcome_spec["family"])
        outcome_df["transform"] = transform
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
            base_sample_df.groupby(["group_label", "group_order"], observed=True)
            .agg(
                actual_transition=("actual_transition", "first"),
                n_obs=("symbol", "size"),
                n_symbols=("symbol", "nunique"),
            )
            .reset_index()
            .sort_values(["group_order", "group_label"], kind="stable")
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
            outcome_df.groupby(["group_label", "group_order"], observed=True)
            .agg(
                outcome=("outcome", "first"),
                display_name=("display_name", "first"),
                side=("side", "first"),
                family=("family", "first"),
                transform=("transform", "first"),
                actual_transition=("actual_transition", "first"),
                n_obs=("symbol", "size"),
                mean_delta=("delta_y", "mean"),
                median_delta=("delta_y", "median"),
                std_delta=("delta_y", "std"),
                mean_y_pre=("y_pre", "mean"),
                mean_y_post=("y_post", "mean"),
            )
            .reset_index()
            .sort_values(["group_order", "group_label"], kind="stable")
            .reset_index(drop=True)
        )

    def fit_group_regression(outcome_df):
        """Estimate group differences in mean outcome changes with OLS.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.

        Returns
        -------
        tuple[RegressionResultsWrapper, pandas.DataFrame]
            OLS fit and group summary table.

        """
        present_groups = set(outcome_df["group_label"].unique().tolist())
        if not set(REQUIRED_GROUPS).issubset(present_groups):
            message = f"All comparison groups are required for regression: {REQUIRED_GROUPS}"
            raise ValueError(message)

        design_df = pd.DataFrame(
            {
                "const": 1.0,
                "to40": outcome_df["treated_to40"].to_numpy(dtype=float),
                "to10": outcome_df["treated_to10"].to_numpy(dtype=float),
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
        group_rows = group_summary_df.set_index("group_label")
        stay100_row = group_rows.loc["stay100"]
        to40_row = group_rows.loc["to40"]
        to10_row = group_rows.loc["to10"]
        coef_to40 = float(fit.params["to40"])
        se_to40 = float(fit.bse["to40"])
        t_to40 = float(fit.tvalues["to40"])
        p_to40 = float(fit.pvalues["to40"])
        ci_to40_low, ci_to40_high = fit.conf_int().loc["to40"].tolist()
        coef_to10 = float(fit.params["to10"])
        se_to10 = float(fit.bse["to10"])
        t_to10 = float(fit.tvalues["to10"])
        p_to10 = float(fit.pvalues["to10"])
        ci_to10_low, ci_to10_high = fit.conf_int().loc["to10"].tolist()

        flat_summary = {
            "status": "ok",
            "outcome": str(outcome_df["outcome"].iloc[0]),
            "display_name": str(outcome_df["display_name"].iloc[0]),
            "side": str(outcome_df["side"].iloc[0]),
            "family": str(outcome_df["family"].iloc[0]),
            "transform": str(outcome_df["transform"].iloc[0]),
            "coef_to40": coef_to40,
            "se_to40": se_to40,
            "t_to40": t_to40,
            "p_value_to40": p_to40,
            "ci_to40_low": float(ci_to40_low),
            "ci_to40_high": float(ci_to40_high),
            "coef_to10": coef_to10,
            "se_to10": se_to10,
            "t_to10": t_to10,
            "p_value_to10": p_to10,
            "ci_to10_low": float(ci_to10_low),
            "ci_to10_high": float(ci_to10_high),
            "n_obs": len(outcome_df),
            "n_stay100": int(stay100_row["n_obs"]),
            "n_to40": int(to40_row["n_obs"]),
            "n_to10": int(to10_row["n_obs"]),
            "r2": float(fit.rsquared),
            "mean_delta_stay100": float(stay100_row["mean_delta"]),
            "mean_delta_to40": float(to40_row["mean_delta"]),
            "mean_delta_to10": float(to10_row["mean_delta"]),
        }
        json_summary = {
            **flat_summary,
            "baseline_group": "stay100",
            "coef_const": float(fit.params["const"]),
            "se_const": float(fit.bse["const"]),
            "group_summary_rows": group_summary_df.to_dict(orient="records"),
        }
        return flat_summary, json_summary

    def make_boxplot(outcome_df, *, outpath):
        """Create a three-group boxplot with jittered raw points.

        Parameters
        ----------
        outcome_df
            Outcome-specific comparison sample.
        outpath
            Output PNG path.

        Returns
        -------
        pathlib.Path
            Saved plot path.

        """
        figure, axis = plt.subplots(figsize=(8, 5))
        ordered_groups = ["stay100", "to40", "to10"]
        display_labels = {
            "stay100": "stay100",
            "to40": "to40",
            "to10": "to10",
        }
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
        box_colors = {
            "stay100": "#93c5fd",
            "to40": "#fca5a5",
            "to10": "#fdba74",
        }
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

        transform = str(outcome_df["transform"].iloc[0])
        ylabel = "log1p(post) - log1p(pre)" if transform == "log1p" else "post - pre"
        axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_title(f"Round-lot comparison: {outcome_df['display_name'].iloc[0]}")
        axis.set_xlabel("Post-reform round-lot group")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(outpath, dpi=200)
        plt.close(figure)
        return outpath

    def make_coefficient_plot(summary_df, *, outcomes, title, xlabel, outpath):
        """Create a coefficient plot for a selected outcome subset.

        Parameters
        ----------
        summary_df
            Aggregate summary table with one row per outcome.
        outcomes
            Ordered outcome names to display.
        title
            Plot title.
        xlabel
            Horizontal axis label.
        outpath
            Output PNG path.

        Returns
        -------
        pathlib.Path
            Saved plot path.

        """
        ok_df = summary_df.loc[
            (summary_df["status"] == "ok") & summary_df["outcome"].isin(list(outcomes))
        ].copy()
        if ok_df.empty:
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.text(0.5, 0.5, "No successful regressions", ha="center", va="center")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(outpath, dpi=200)
            plt.close(figure)
            return outpath

        outcome_order = {outcome_name: index for index, outcome_name in enumerate(outcomes)}
        ok_df["plot_order"] = ok_df["outcome"].map(outcome_order)
        ok_df = ok_df.sort_values("plot_order", kind="stable").reset_index(drop=True)
        y_positions = np.arange(len(ok_df))
        lower_errors_to40 = ok_df["coef_to40"] - ok_df["ci_to40_low"]
        upper_errors_to40 = ok_df["ci_to40_high"] - ok_df["coef_to40"]
        lower_errors_to10 = ok_df["coef_to10"] - ok_df["ci_to10_low"]
        upper_errors_to10 = ok_df["ci_to10_high"] - ok_df["coef_to10"]
        offset = 0.12 if len(ok_df) > 1 else 0.08

        figure, axis = plt.subplots(figsize=(9, max(4.0, 0.8 * len(ok_df) + 1.4)))
        axis.errorbar(
            ok_df["coef_to40"],
            y_positions + offset,
            xerr=[lower_errors_to40, upper_errors_to40],
            fmt="o",
            color="#1d4ed8",
            ecolor="#60a5fa",
            elinewidth=1.5,
            capsize=3,
            label="to40 - stay100",
        )
        axis.errorbar(
            ok_df["coef_to10"],
            y_positions - offset,
            xerr=[lower_errors_to10, upper_errors_to10],
            fmt="o",
            color="#c2410c",
            ecolor="#fdba74",
            elinewidth=1.5,
            capsize=3,
            label="to10 - stay100",
        )
        axis.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(ok_df["display_name"].tolist())
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(alpha=0.2, axis="x")
        axis.legend(frameon=False)
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
        derive_aggregated_outcomes,
        fit_group_regression,
        load_round_lot_panel,
        make_boxplot,
        make_coefficient_plot,
        summarize_base_groups,
        summarize_regression,
        write_json,
    )


@app.cell
def _(component_definition_df, mo):
    outcome_construction_section = mo.md(
        """
        ## Outcome Construction

        This section maps each raw cross-K window to a distance-from-zero band,
        then defines a side-specific total and three within-side shares for each
        sign. It also carries the direct mode-location outcome
        `closest_mode_to_zero_sec -> mode_position_sec`.
        """
    )
    outcome_construction_section
    component_definition_df
    return


@app.cell
def _(
    POST_RL_COL,
    PRE_RL_COL,
    allowed_post_round_lots,
    build_base_sample,
    build_cross_k_panel,
    component_groups,
    date_post,
    date_pre,
    derive_aggregated_outcomes,
    direct_outcome_specs,
    group_labels,
    load_round_lot_panel,
    outcome_specs,
    outcome_specs_df,
    pd,
    pre_round_lot_required,
    quote_exchange,
    raw_panel_columns,
    resolved_input_csv_path,
    sum_outcomes,
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

    raw_panel_df = build_cross_k_panel(
        raw_summary_df,
        trade_exchange=trade_exchange,
        quote_exchange=quote_exchange,
        date_pre=date_pre,
        date_post=date_post,
        raw_columns=raw_panel_columns,
    )
    aggregated_panel_df = derive_aggregated_outcomes(
        raw_panel_df,
        component_groups=component_groups,
        date_pre=date_pre,
        date_post=date_post,
        direct_outcome_specs=direct_outcome_specs,
        sum_outcomes=sum_outcomes,
    )
    round_lot_panel_df = load_round_lot_panel(date_pre=date_pre, date_post=date_post)
    panel_df = aggregated_panel_df.merge(round_lot_panel_df, on="symbol", how="left")

    available_y_stems = [str(spec["outcome"]) for spec in outcome_specs]
    outcome_specs_by_name = {
        str(spec["outcome"]): {
            "display_name": str(spec["display_name"]),
            "side": str(spec["side"]),
            "family": str(spec["family"]),
            "transform": str(spec["transform"]),
        }
        for spec in outcome_specs
    }
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

    outcome_sample_size_rows = []
    for _outcome_name in available_y_stems:
        pre_y_col = f"{_outcome_name}_{date_pre}"
        post_y_col = f"{_outcome_name}_{date_post}"
        _outcome_df = base_sample_df.loc[
            base_sample_df[pre_y_col].notna() & base_sample_df[post_y_col].notna(),
            [pre_y_col, post_y_col, "group_label"],
        ].copy()
        _group_counts = _outcome_df["group_label"].value_counts(dropna=False).to_dict()
        outcome_sample_size_rows.append(
            {
                "outcome": _outcome_name,
                "display_name": outcome_specs_by_name[_outcome_name]["display_name"],
                "side": outcome_specs_by_name[_outcome_name]["side"],
                "family": outcome_specs_by_name[_outcome_name]["family"],
                "n_obs_nonmissing": len(_outcome_df),
                "n_stay100_nonmissing": int(_group_counts.get(group_labels[0], 0)),
                "n_to40_nonmissing": int(_group_counts.get(group_labels[1], 0)),
                "n_to10_nonmissing": int(_group_counts.get(group_labels[2], 0)),
            }
        )
    outcome_sample_size_df = pd.DataFrame(outcome_sample_size_rows)
    outcome_list_df = outcome_specs_df.copy()
    return (
        available_y_stems,
        base_sample_df,
        excluded_post_round_lot_counts,
        outcome_list_df,
        outcome_sample_size_df,
        outcome_specs_by_name,
        pair_ok_df,
        panel_df,
        raw_summary_df,
        round_lot_panel_df,
    )


@app.cell
def _(allowed_post_round_lots, base_sample_df, mo):
    sample_construction_section = mo.md(
        f"""
        ## Sample Construction

        This section builds the observed comparison sample. The base sample keeps
        symbols with `round_lot_20251031 == 100` and post-reform round lots in
        `{allowed_post_round_lots}`, then defines `stay100`, `to40`, and `to10`
        groups.

        - `n_base_sample`: `{len(base_sample_df)}`
        """
    )
    sample_construction_section
    return


@app.cell
def _(base_sample_df, summarize_base_groups):
    base_group_summary_df = summarize_base_groups(base_sample_df)
    return (base_group_summary_df,)


@app.cell
def _(
    Path,
    UTC,
    allowed_post_round_lots,
    analysis_note,
    available_y_stems,
    base_sample_df,
    build_outcome_sample,
    coefficient_plot_specs,
    date_post,
    date_pre,
    datetime,
    excluded_post_round_lot_counts,
    export_analysis_sample,
    export_group_summary,
    fit_group_regression,
    group_labels,
    input_csv_stem,
    make_boxplot,
    make_coefficient_plot,
    outcome_specs_by_name,
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
    coefficient_plot_paths = {}
    base_transition_counts = (
        base_sample_df["actual_transition"].value_counts(dropna=False).sort_index().to_dict()
    )

    for _outcome_name in available_y_stems:
        _outcome_spec = outcome_specs_by_name[_outcome_name]
        summary_path = resolved_output_dir / f"regression__{_outcome_name}.json"
        try:
            _outcome_df = build_outcome_sample(
                base_sample_df,
                outcome_name=_outcome_name,
                outcome_specs_by_name=outcome_specs_by_name,
            )
            fit, group_summary_df = fit_group_regression(_outcome_df)
            flat_summary, json_summary = summarize_regression(
                _outcome_df,
                fit=fit,
                group_summary_df=group_summary_df,
            )

            analysis_sample_path = None
            if export_analysis_sample:
                analysis_sample_path = resolved_output_dir / f"analysis_sample__{_outcome_name}.csv"
                _outcome_df.to_csv(analysis_sample_path, index=False)
                artifact_rows.append(
                    {
                        "outcome": _outcome_name,
                        "artifact_type": "analysis_sample_csv",
                        "path": str(analysis_sample_path),
                    }
                )

            group_summary_path = None
            if export_group_summary:
                group_summary_path = resolved_output_dir / f"group_summary__{_outcome_name}.csv"
                group_summary_df.to_csv(group_summary_path, index=False)
                artifact_rows.append(
                    {
                        "outcome": _outcome_name,
                        "artifact_type": "group_summary_csv",
                        "path": str(group_summary_path),
                    }
                )

            boxplot_path = resolved_output_dir / f"boxplot__{_outcome_name}.png"
            make_boxplot(_outcome_df, outpath=boxplot_path)
            artifact_rows.append(
                {
                    "outcome": _outcome_name,
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
                    "outcome": _outcome_name,
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
                "outcome": _outcome_name,
                "display_name": _outcome_spec["display_name"],
                "side": _outcome_spec["side"],
                "family": _outcome_spec["family"],
                "transform": _outcome_spec["transform"],
                "coef_to40": None,
                "se_to40": None,
                "t_to40": None,
                "p_value_to40": None,
                "ci_to40_low": None,
                "ci_to40_high": None,
                "coef_to10": None,
                "se_to10": None,
                "t_to10": None,
                "p_value_to10": None,
                "ci_to10_low": None,
                "ci_to10_high": None,
                "n_obs": 0,
                "n_stay100": 0,
                "n_to40": 0,
                "n_to10": 0,
                "r2": None,
                "mean_delta_stay100": None,
                "mean_delta_to40": None,
                "mean_delta_to10": None,
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
                    "outcome": _outcome_name,
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

    for plot_spec in coefficient_plot_specs:
        coefficient_plot_path = resolved_output_dir / str(plot_spec["filename"])
        make_coefficient_plot(
            analysis_results_df,
            outcomes=tuple(plot_spec["outcomes"]),
            title=str(plot_spec["title"]),
            xlabel=str(plot_spec["xlabel"]),
            outpath=coefficient_plot_path,
        )
        coefficient_plot_paths[str(plot_spec["plot_key"])] = coefficient_plot_path
        artifact_rows.append(
            {
                "outcome": "ALL",
                "artifact_type": f"coef_plot_png__{plot_spec['plot_key']}",
                "path": str(coefficient_plot_path),
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
        "pre_round_lot_required": pre_round_lot_required,
        "allowed_post_round_lots": list(allowed_post_round_lots),
        "group_labels": list(group_labels),
        "available_outcomes": list(available_y_stems),
        "n_raw_rows": len(raw_summary_df),
        "n_pair_ok_rows": len(pair_ok_df),
        "n_panel_symbols": int(panel_df["symbol"].nunique()),
        "n_round_lot_symbols": int(round_lot_panel_df["symbol"].nunique()),
        "n_base_sample": len(base_sample_df),
        "base_transition_counts": base_transition_counts,
        "n_successful_outcomes": int((analysis_results_df["status"] == "ok").sum()),
        "excluded_post_round_lot_counts": excluded_post_round_lot_counts,
        "coefficient_plot_paths": {key: str(path) for key, path in coefficient_plot_paths.items()},
        "artifact_files": artifact_manifest_df.to_dict(orient="records"),
        "analysis_note": analysis_note,
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
                "group_labels": ", ".join(group_labels),
                "n_outcomes": len(available_y_stems),
            }
        ]
    )
    return (
        analysis_results_df,
        artifact_manifest_df,
        coefficient_plot_paths,
        input_overview_df,
        run_manifest,
    )


@app.cell
def _(mo):
    results_section = mo.md(
        """
        ## Results

        This section reports the overall input coverage, the base comparison
        groups, the nonmissing outcome counts, and the regression summary for the
        eight aggregated outcomes.
        """
    )
    results_section
    return


@app.cell
def _(input_overview_df):
    input_overview_df
    return


@app.cell
def _(outcome_list_df):
    outcome_list_df
    return


@app.cell
def _(base_group_summary_df):
    base_group_summary_df
    return


@app.cell
def _(outcome_sample_size_df):
    outcome_sample_size_df
    return


@app.cell
def _(analysis_results_df):
    analysis_results_df
    return


@app.cell
def _(mo):
    significance_section = mo.md(
        """
        ## Significance Plots

        This section shows four coefficient plots. Each plot overlays the
        `to40 - stay100` and `to10 - stay100` estimates for the requested
        outcome family.
        """
    )
    significance_section
    return


@app.cell
def _(coefficient_plot_paths, mpimg, plt):
    neg_sum_plot_path = coefficient_plot_paths["negative_total"]
    neg_sum_plot_image = mpimg.imread(neg_sum_plot_path)
    neg_sum_plot_preview, _axis = plt.subplots(figsize=(9, 4.5))
    _axis.imshow(neg_sum_plot_image)
    _axis.axis("off")
    _axis.set_title(str(neg_sum_plot_path))
    neg_sum_plot_preview.tight_layout()
    neg_sum_plot_preview
    return


@app.cell
def _(coefficient_plot_paths, mpimg, plt):
    pos_sum_plot_path = coefficient_plot_paths["positive_total"]
    pos_sum_plot_image = mpimg.imread(pos_sum_plot_path)
    pos_sum_plot_preview, _axis = plt.subplots(figsize=(9, 4.5))
    _axis.imshow(pos_sum_plot_image)
    _axis.axis("off")
    _axis.set_title(str(pos_sum_plot_path))
    pos_sum_plot_preview.tight_layout()
    pos_sum_plot_preview
    return


@app.cell
def _(coefficient_plot_paths, mpimg, plt):
    neg_shares_plot_path = coefficient_plot_paths["negative_shares"]
    neg_shares_plot_image = mpimg.imread(neg_shares_plot_path)
    neg_shares_plot_preview, _axis = plt.subplots(figsize=(9, 5.0))
    _axis.imshow(neg_shares_plot_image)
    _axis.axis("off")
    _axis.set_title(str(neg_shares_plot_path))
    neg_shares_plot_preview.tight_layout()
    neg_shares_plot_preview
    return


@app.cell
def _(coefficient_plot_paths, mpimg, plt):
    pos_shares_plot_path = coefficient_plot_paths["positive_shares"]
    pos_shares_plot_image = mpimg.imread(pos_shares_plot_path)
    pos_shares_plot_preview, _axis = plt.subplots(figsize=(9, 5.0))
    _axis.imshow(pos_shares_plot_image)
    _axis.axis("off")
    _axis.set_title(str(pos_shares_plot_path))
    pos_shares_plot_preview.tight_layout()
    pos_shares_plot_preview
    return


@app.cell
def _(mo):
    artifacts_section = mo.md(
        """
        ## Artifacts

        This section lists every written artifact and the run manifest that ties
        the outputs back to the exact notebook configuration.
        """
    )
    artifacts_section
    return


@app.cell
def _(artifact_manifest_df):
    artifact_manifest_df
    return


@app.cell
def _(run_manifest):
    run_manifest
    return


if __name__ == "__main__":
    app.run()
