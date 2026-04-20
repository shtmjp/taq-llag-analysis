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
def _(Path, pd):
    analysis_note = "Observed price-threshold comparison only; not a causal RD design."
    input_csv_path = Path("../data/derived/trade_quote_modes/my-run/cross_k_summary.csv")
    cta_symbol_dir = Path("../data/cta_symbol_files_202509")
    output_base_dir = Path("output/cross_k_sum_share_rd_250")
    trade_exchange = "N"
    quote_exchange = "P"
    date_pre = "20251031"
    date_post = "20251103"
    price_column = "ConsolidatedClosingPrice"
    cutoff = 250.0
    upper_bound = 1000.0
    pre_round_lot_required = 100.0
    enforce_rule_consistency = True
    bins_per_side = 10
    export_analysis_sample = True
    export_balance_tests = True
    balance_covariates = None
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
    return (
        analysis_note,
        balance_covariates,
        bins_per_side,
        component_definition_df,
        component_groups,
        cta_symbol_dir,
        cutoff,
        date_post,
        date_pre,
        direct_outcome_specs,
        enforce_rule_consistency,
        export_analysis_sample,
        export_balance_tests,
        input_csv_path,
        outcome_specs,
        outcome_specs_df,
        output_base_dir,
        pre_round_lot_required,
        price_column,
        quote_exchange,
        raw_panel_columns,
        sum_outcomes,
        trade_exchange,
        upper_bound,
    )


@app.cell
def _(analysis_note, mo):
    overview_section = mo.md(
        f"""
        ## Overview

        This notebook estimates RD effects at the $250 round-lot threshold for
        the `N/P` exchange pair using September CTA close prices as the running
        variable. Outcomes are the aggregated cross-K totals and shares plus the
        direct mode-position outcome defined in the companion sum/share notebook.

        `{analysis_note}`
        """
    )
    overview_section
    return


@app.cell
def _(
    bins_per_side,
    cta_symbol_dir,
    cutoff,
    date_post,
    date_pre,
    enforce_rule_consistency,
    input_csv_path,
    mo,
    output_base_dir,
    pre_round_lot_required,
    price_column,
    quote_exchange,
    trade_exchange,
    upper_bound,
):
    inputs_section = mo.md(
        f"""
        ## Inputs

        This section records the raw cross-K source, CTA price source, exchange
        pair, and RD tuning parameters.

        - `input_csv_path`: `{input_csv_path}`
        - `cta_symbol_dir`: `{cta_symbol_dir}`
        - `output_base_dir`: `{output_base_dir}`
        - `trade_exchange`: `{trade_exchange}`
        - `quote_exchange`: `{quote_exchange}`
        - `date_pre`: `{date_pre}`
        - `date_post`: `{date_post}`
        - `price_column`: `{price_column}`
        - `cutoff`: `{cutoff}`
        - `upper_bound`: `{upper_bound}`
        - `pre_round_lot_required`: `{pre_round_lot_required}`
        - `enforce_rule_consistency`: `{enforce_rule_consistency}`
        - `bins_per_side`: `{bins_per_side}`
        """
    )
    inputs_section
    return


@app.cell
def _(
    Path,
    cta_symbol_dir,
    input_csv_path,
    output_base_dir,
    quote_exchange,
    trade_exchange,
):
    resolved_input_csv_path = Path(input_csv_path)
    if not resolved_input_csv_path.exists():
        message = f"cross_k summary not found: {resolved_input_csv_path}"
        raise FileNotFoundError(message)

    resolved_cta_symbol_dir = Path(cta_symbol_dir)
    if not resolved_cta_symbol_dir.exists():
        message = f"CTA symbol dir not found: {resolved_cta_symbol_dir}"
        raise FileNotFoundError(message)

    cta_price_files = sorted(resolved_cta_symbol_dir.glob("CTA.Symbol.File.202509*.csv"))
    if not cta_price_files:
        message = f"No CTA symbol files found under {resolved_cta_symbol_dir}"
        raise FileNotFoundError(message)

    input_csv_stem = resolved_input_csv_path.stem
    pair_label = f"{trade_exchange}{quote_exchange}"
    resolved_output_dir = output_base_dir / f"{input_csv_stem}_{pair_label}"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return (
        cta_price_files,
        input_csv_stem,
        pair_label,
        resolved_cta_symbol_dir,
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
    DATE_PRE = date_pre
    DATE_POST = date_post
    PRICE_COL = "mean_close_202509"
    PRE_RL_COL = f"round_lot_{DATE_PRE}"
    POST_RL_COL = f"round_lot_{DATE_POST}"
    PRE_REFORM_ROUND_LOT = 100.0
    POST_REFORM_SMALL_LOT = 40.0
    DENSITY_STATUS = "omitted"
    DENSITY_REASON = "rddensity is incompatible with pandas 3 in the current environment."
    MIN_SIDE_OBS = 2
    MIN_BIN_EDGES = 2
    MIN_BALANCE_OBS = 10
    MIN_ANALYSIS_OBS = 20

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

    def load_cta_mean_close_panel(*, cta_price_files, price_column):
        """Build a symbol-level September mean close table from CTA files.

        Parameters
        ----------
        cta_price_files
            Daily CTA symbol-file paths for September 2025.
        price_column
            CTA price column name to average.

        Returns
        -------
        pandas.DataFrame
            One row per symbol with the September mean close.

        """
        price_frames = []
        for path in cta_price_files:
            price_df = pd.read_csv(
                path,
                usecols=["Symbol", price_column],
            ).rename(
                columns={
                    "Symbol": "symbol",
                    price_column: "close_price",
                }
            )
            price_frames.append(price_df)

        combined_df = pd.concat(price_frames, ignore_index=True)
        return (
            combined_df.groupby("symbol", as_index=False)
            .agg(
                mean_close_202509=("close_price", "mean"),
                n_close_obs_202509=("close_price", "count"),
            )
            .sort_values("symbol", kind="stable")
            .reset_index(drop=True)
        )

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
                }
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
            }
        )
        wide_round_lot_df.index.name = "symbol"
        return wide_round_lot_df.sort_index().reset_index()

    def make_delta(pre, post, transform):
        """Construct the outcome change used in RD estimation.

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
            Outcome change series.

        """
        if transform == "signed":
            return post - pre
        if transform == "log1p":
            return np.log1p(post) - np.log1p(pre)
        message = f"Unknown transform: {transform!r}"
        raise ValueError(message)

    def expected_round_lot_250(price, cutoff):
        """Compute the expected post-reform round lot at the $250 rule.

        Parameters
        ----------
        price
            September mean close.
        cutoff
            RD cutoff in dollars.

        Returns
        -------
        pandas.Series
            Expected round lot under the reform rule.

        """
        return pd.Series(
            np.where(price <= cutoff, PRE_REFORM_ROUND_LOT, POST_REFORM_SMALL_LOT),
            index=price.index,
        )

    def build_rd_sample(
        df,
        y_stem,
        *,
        cutoff,
        upper_bound,
        enforce_rule_consistency,
        outcome_specs_by_name,
        pre_round_lot_required,
    ):
        """Build the sharp RD sample for one outcome.

        Parameters
        ----------
        df
            Raw panel dataframe.
        y_stem
            Outcome stem.
        cutoff
            RD cutoff in dollars.
        upper_bound
            Upper support cutoff for September mean close.
        enforce_rule_consistency
            Whether to keep only rows consistent with the post-reform rule.
        outcome_specs_by_name
            Outcome metadata keyed by outcome name.
        pre_round_lot_required
            Required pre-reform round lot.

        Returns
        -------
        pandas.DataFrame
            Analysis sample with derived RD columns.

        """
        pre_y_col = f"{y_stem}_{date_pre}"
        post_y_col = f"{y_stem}_{date_post}"
        required_columns = [
            "symbol",
            PRICE_COL,
            PRE_RL_COL,
            POST_RL_COL,
            pre_y_col,
            post_y_col,
        ]
        missing_columns = [
            column_name for column_name in required_columns if column_name not in df.columns
        ]
        if missing_columns:
            message = f"Missing required columns: {missing_columns}"
            raise KeyError(message)

        transform = str(outcome_specs_by_name[y_stem]["transform"])
        analysis_df = df.copy()
        analysis_df = analysis_df.loc[analysis_df[PRICE_COL].notna()].copy()
        analysis_df = analysis_df.loc[analysis_df[PRICE_COL] < upper_bound].copy()
        analysis_df = analysis_df.loc[analysis_df[PRE_RL_COL] == pre_round_lot_required].copy()
        analysis_df = analysis_df.loc[analysis_df[POST_RL_COL].notna()].copy()

        analysis_df["y_pre"] = pd.to_numeric(analysis_df[pre_y_col], errors="coerce")
        analysis_df["y_post"] = pd.to_numeric(analysis_df[post_y_col], errors="coerce")
        analysis_df["delta_y"] = make_delta(
            analysis_df["y_pre"],
            analysis_df["y_post"],
            transform=transform,
        )
        analysis_df = analysis_df.loc[analysis_df["delta_y"].notna()].copy()

        analysis_df["display_name"] = str(outcome_specs_by_name[y_stem]["display_name"])
        analysis_df["side"] = str(outcome_specs_by_name[y_stem]["side"])
        analysis_df["family"] = str(outcome_specs_by_name[y_stem]["family"])
        analysis_df["transform"] = transform
        analysis_df["running"] = analysis_df[PRICE_COL] - cutoff
        analysis_df["D_rule"] = (analysis_df["running"] > 0).astype(int)
        analysis_df["expected_post_round_lot"] = expected_round_lot_250(
            analysis_df[PRICE_COL],
            cutoff=cutoff,
        )
        analysis_df["rule_consistent"] = (
            analysis_df[POST_RL_COL] == analysis_df["expected_post_round_lot"]
        )
        analysis_df["D_actual"] = (analysis_df[POST_RL_COL] == POST_REFORM_SMALL_LOT).astype(int)
        if enforce_rule_consistency:
            analysis_df = analysis_df.loc[analysis_df["rule_consistent"]].copy()

        return analysis_df

    def triangular_weights(x_centered, h_left, h_right):
        """Compute asymmetric triangular RD weights.

        Parameters
        ----------
        x_centered
            Running variable centered at the cutoff.
        h_left
            Left-side bandwidth.
        h_right
            Right-side bandwidth.

        Returns
        -------
        numpy.ndarray
            Nonnegative local-linear weights.

        """
        centered = np.asarray(x_centered, dtype=float)
        weights = np.zeros_like(centered, dtype=float)
        left_mask = centered < 0
        right_mask = ~left_mask
        weights[left_mask] = np.maximum(1.0 - np.abs(centered[left_mask]) / h_left, 0.0)
        weights[right_mask] = np.maximum(1.0 - np.abs(centered[right_mask]) / h_right, 0.0)
        return weights

    def fallback_bandwidths_from_k(
        x,
        cutoff,
        *,
        k_left=10,
        k_right=6,
        b_multiplier=1.5,
    ):
        """Build deterministic fallback bandwidths from nearest distances.

        Parameters
        ----------
        x
            Running-variable support in dollars.
        cutoff
            RD cutoff in dollars.
        k_left
            Rank used on the left.
        k_right
            Rank used on the right.
        b_multiplier
            Bias bandwidth multiplier.

        Returns
        -------
        dict[str, float | int]
            Left/right estimation and bias bandwidths.

        """
        running = np.asarray(x, dtype=float)
        left_dist = np.sort(cutoff - running[running <= cutoff])
        right_dist = np.sort(running[running > cutoff] - cutoff)
        if len(left_dist) < MIN_SIDE_OBS or len(right_dist) < MIN_SIDE_OBS:
            message = "Need at least two observations on each side for fallback bandwidths."
            raise ValueError(message)

        h_left = float(left_dist[min(k_left, len(left_dist)) - 1])
        h_right = float(right_dist[min(k_right, len(right_dist)) - 1])
        h_left = max(h_left, 1e-6)
        h_right = max(h_right, 1e-6)
        return {
            "h_left": h_left,
            "h_right": h_right,
            "b_left": max(b_multiplier * h_left, h_left + 1e-6),
            "b_right": max(b_multiplier * h_right, h_right + 1e-6),
            "k_left_used": int(min(k_left, len(left_dist))),
            "k_right_used": int(min(k_right, len(right_dist))),
        }

    def manual_local_linear_fit(y, x, cutoff, h_left, h_right):
        """Fit a local-linear RD regression for plotting.

        Parameters
        ----------
        y
            Outcome change.
        x
            Running variable in dollar units.
        cutoff
            RD cutoff.
        h_left
            Left bandwidth.
        h_right
            Right bandwidth.

        Returns
        -------
        tuple[RegressionResultsWrapper, numpy.ndarray]
            Weighted least-squares fit and keep mask.

        """
        x_centered = np.asarray(x - cutoff, dtype=float)
        y_array = np.asarray(y, dtype=float)
        treatment = (x_centered > 0).astype(float)
        weights = triangular_weights(x_centered, h_left=h_left, h_right=h_right)
        keep_mask = weights > 0
        design = np.column_stack(
            [
                np.ones(keep_mask.sum()),
                treatment[keep_mask],
                x_centered[keep_mask],
                treatment[keep_mask] * x_centered[keep_mask],
            ]
        )
        fit = sm.WLS(y_array[keep_mask], design, weights=weights[keep_mask]).fit(cov_type="HC1")
        return fit, keep_mask

    def make_manual_rd_plot(
        df,
        *,
        cutoff,
        outpath,
        y_label,
        title,
        bins_per_side,
        h_left=None,
        h_right=None,
    ):
        """Create and save a binned RD plot with local-linear overlays.

        Parameters
        ----------
        df
            RD analysis sample.
        cutoff
            RD cutoff in dollars.
        outpath
            Output PNG path.
        y_label
            Y-axis label.
        title
            Plot title.
        bins_per_side
            Number of bins on each side.
        h_left
            Optional left bandwidth for plotting.
        h_right
            Optional right bandwidth for plotting.

        Returns
        -------
        pathlib.Path
            Saved plot path.

        """
        x = df[PRICE_COL].to_numpy(dtype=float)
        y = df["delta_y"].to_numpy(dtype=float)

        if h_left is None or h_right is None:
            x_centered = x - cutoff
            left_span = np.abs(x_centered[x_centered < 0])
            right_span = np.abs(x_centered[x_centered > 0])
            h_left = np.nanpercentile(left_span, 75) if len(left_span) else 25.0
            h_right = np.nanpercentile(right_span, 75) if len(right_span) else 25.0
            h_left = max(float(h_left), 10.0)
            h_right = max(float(h_right), 10.0)

        fit, keep_mask = manual_local_linear_fit(
            y=y,
            x=x,
            cutoff=cutoff,
            h_left=h_left,
            h_right=h_right,
        )
        plot_df = df.loc[keep_mask, [PRICE_COL, "delta_y"]].copy()
        left_df = plot_df.loc[plot_df[PRICE_COL] <= cutoff].copy()
        right_df = plot_df.loc[plot_df[PRICE_COL] > cutoff].copy()

        def binned_means(side_df, side):
            if side_df.empty:
                return pd.DataFrame(columns=["x", "y"])
            if side == "left":
                bins = np.linspace(side_df[PRICE_COL].min(), cutoff, bins_per_side + 1)
            else:
                bins = np.linspace(cutoff, side_df[PRICE_COL].max(), bins_per_side + 1)
            bins = np.unique(bins)
            if len(bins) < MIN_BIN_EDGES:
                return pd.DataFrame(
                    {
                        "x": [side_df[PRICE_COL].mean()],
                        "y": [side_df["delta_y"].mean()],
                    }
                )
            grouped_df = side_df.assign(
                bin=pd.cut(
                    side_df[PRICE_COL],
                    bins=bins,
                    include_lowest=True,
                    duplicates="drop",
                )
            )
            return (
                grouped_df.groupby("bin", observed=True)
                .agg(x=(PRICE_COL, "mean"), y=("delta_y", "mean"))
                .reset_index(drop=True)
            )

        left_bins = binned_means(left_df, "left")
        right_bins = binned_means(right_df, "right")

        x_left = np.linspace(cutoff - h_left, cutoff, 200)
        x_right = np.linspace(cutoff, cutoff + h_right, 200)
        beta = fit.params
        y_left = beta[0] + beta[2] * (x_left - cutoff)
        y_right = beta[0] + beta[1] + (beta[2] + beta[3]) * (x_right - cutoff)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(left_bins["x"], left_bins["y"], label="Left bins", color="#1d4ed8")
        axis.scatter(right_bins["x"], right_bins["y"], label="Right bins", color="#b91c1c")
        axis.plot(x_left, y_left, linewidth=2, color="#1d4ed8")
        axis.plot(x_right, y_right, linewidth=2, color="#b91c1c")
        axis.axvline(cutoff, linestyle="--", linewidth=1, color="#111827")
        axis.set_title(title)
        axis.set_xlabel("September mean close")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(outpath, dpi=200)
        plt.close(figure)
        return outpath

    def try_rdrobust(*, y, x, cutoff, covs=None):
        """Run rdrobust with automatic bandwidths and deterministic fallback.

        Parameters
        ----------
        y
            RD outcome array.
        x
            Running variable array.
        cutoff
            RD cutoff in dollars.
        covs
            Optional predetermined covariate dataframe.

        Returns
        -------
        tuple[object | None, object, dict[str, object]]
            Automatic bandwidth object, rdrobust result, and metadata.

        """
        from rdrobust import rdbwselect, rdrobust

        est_kwargs = {
            "y": y,
            "x": x,
            "c": cutoff,
            "p": 1,
            "q": 2,
            "kernel": "tri",
            "bwselect": "mserd",
        }
        if covs is not None and not covs.empty:
            est_kwargs["covs"] = covs

        try:
            bw = rdbwselect(
                y=y,
                x=x,
                c=cutoff,
                p=1,
                q=2,
                kernel="tri",
                bwselect="mserd",
            )
            est = rdrobust(**est_kwargs)
            meta = {
                "bandwidth_source": "automatic",
                "fallback_info": None,
                "auto_error": None,
            }
        except Exception as exc:  # noqa: BLE001
            fallback_info = fallback_bandwidths_from_k(x=x, cutoff=cutoff)
            est_kwargs.pop("bwselect", None)
            est_kwargs["h"] = [fallback_info["h_left"], fallback_info["h_right"]]
            est_kwargs["b"] = [fallback_info["b_left"], fallback_info["b_right"]]
            est = rdrobust(**est_kwargs)
            meta = {
                "bandwidth_source": "manual_fallback",
                "fallback_info": fallback_info,
                "auto_error": repr(exc),
            }
            return None, est, meta
        return bw, est, meta

    def summarize_rdrobust(est):
        """Extract a compact rdrobust summary.

        Parameters
        ----------
        est
            rdrobust result object.

        Returns
        -------
        dict[str, object]
            Compact summary with coefficients, confidence intervals, p-values,
            and bandwidths.

        """
        summary = {
            "coef_conventional": None,
            "coef_bias_corrected": None,
            "coef_robust": None,
            "ci_conventional": None,
            "ci_bias_corrected": None,
            "ci_robust": None,
            "p_conventional": None,
            "p_bias_corrected": None,
            "p_robust": None,
            "bandwidth_h_left": None,
            "bandwidth_h_right": None,
            "bandwidth_b_left": None,
            "bandwidth_b_right": None,
        }

        estimate_table = est.Estimate.copy()
        summary["coef_conventional"] = float(estimate_table.iloc[0]["tau.us"])
        summary["coef_bias_corrected"] = float(estimate_table.iloc[0]["tau.bc"])
        summary["coef_robust"] = float(estimate_table.iloc[0]["tau.bc"])

        ci_table = est.ci.copy()
        summary["ci_conventional"] = [float(ci_table.iloc[0, 0]), float(ci_table.iloc[0, 1])]
        summary["ci_bias_corrected"] = [float(ci_table.iloc[1, 0]), float(ci_table.iloc[1, 1])]
        summary["ci_robust"] = [float(ci_table.iloc[2, 0]), float(ci_table.iloc[2, 1])]

        pv_table = est.pv.copy()
        summary["p_conventional"] = float(pv_table.iloc[0, 0])
        summary["p_bias_corrected"] = float(pv_table.iloc[1, 0])
        summary["p_robust"] = float(pv_table.iloc[2, 0])

        bandwidth_table = est.bws.copy()
        summary["bandwidth_h_left"] = float(bandwidth_table.loc["h", "left"])
        summary["bandwidth_h_right"] = float(bandwidth_table.loc["h", "right"])
        summary["bandwidth_b_left"] = float(bandwidth_table.loc["b", "left"])
        summary["bandwidth_b_right"] = float(bandwidth_table.loc["b", "right"])
        return summary

    def default_balance_covariates(df, y_stem):
        """Infer predetermined covariates for balance tests.

        Parameters
        ----------
        df
            Raw panel dataframe.
        y_stem
            Outcome stem whose own pre column should be excluded.

        Returns
        -------
        list[str]
            Sorted pre-reform covariate column names.

        """
        pre_suffix = f"_{date_pre}"
        current_pre = f"{y_stem}{pre_suffix}"
        return sorted(
            column_name
            for column_name in df.columns
            if str(column_name).endswith(pre_suffix)
            and column_name not in {current_pre, PRE_RL_COL}
        )

    def run_balance_tests(df, covariates, *, cutoff):
        """Run rdrobust balance tests for predetermined covariates.

        Parameters
        ----------
        df
            Analysis sample dataframe.
        covariates
            Predetermined covariate column names.
        cutoff
            RD cutoff in dollars.

        Returns
        -------
        pandas.DataFrame
            One-row-per-covariate balance summary.

        """
        rows = []
        for covariate in covariates:
            if covariate not in df.columns:
                continue
            keep_mask = df[covariate].notna()
            if int(keep_mask.sum()) < MIN_BALANCE_OBS:
                continue
            _, est, _meta = try_rdrobust(
                y=df.loc[keep_mask, covariate].to_numpy(dtype=float),
                x=df.loc[keep_mask, PRICE_COL].to_numpy(dtype=float),
                cutoff=cutoff,
            )
            cov_summary = summarize_rdrobust(est)
            rows.append(
                {
                    "covariate": covariate,
                    "rd_effect_robust": cov_summary["coef_robust"],
                    "p_value_robust": cov_summary["p_robust"],
                    "ci_robust_low": cov_summary["ci_robust"][0],
                    "ci_robust_high": cov_summary["ci_robust"][1],
                }
            )
        return pd.DataFrame(rows)

    def json_default(value):
        """Convert notebook objects into JSON-safe representations.

        Parameters
        ----------
        value
            Object passed to `json.dump`.

        Returns
        -------
        object
            JSON-safe replacement.

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
        DATE_POST,
        DATE_PRE,
        DENSITY_REASON,
        DENSITY_STATUS,
        MIN_ANALYSIS_OBS,
        PRICE_COL,
        build_cross_k_panel,
        build_rd_sample,
        default_balance_covariates,
        derive_aggregated_outcomes,
        load_cta_mean_close_panel,
        load_round_lot_panel,
        make_manual_rd_plot,
        run_balance_tests,
        summarize_rdrobust,
        try_rdrobust,
        write_json,
    )


@app.cell
def _(mo):
    cta_price_section = mo.md(
        """
        ## CTA Price Construction

        This section builds the running variable by averaging the daily
        `ConsolidatedClosingPrice` entries from the September 2025 CTA symbol
        files, then joining that symbol-level mean to the N/P cross-K panel.
        """
    )
    cta_price_section
    return


@app.cell
def _(
    PRICE_COL,
    build_cross_k_panel,
    component_groups,
    cta_price_files,
    date_post,
    date_pre,
    derive_aggregated_outcomes,
    direct_outcome_specs,
    input_csv_stem,
    load_cta_mean_close_panel,
    load_round_lot_panel,
    outcome_specs,
    outcome_specs_df,
    pd,
    price_column,
    quote_exchange,
    raw_panel_columns,
    resolved_cta_symbol_dir,
    resolved_input_csv_path,
    resolved_output_dir,
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
    cta_mean_close_df = load_cta_mean_close_panel(
        cta_price_files=cta_price_files,
        price_column=price_column,
    )
    round_lot_panel_df = load_round_lot_panel(date_pre=date_pre, date_post=date_post)

    panel_df = (
        aggregated_panel_df.merge(cta_mean_close_df, on="symbol", how="left")
        .merge(round_lot_panel_df, on="symbol", how="left")
        .sort_values("symbol", kind="stable")
        .reset_index(drop=True)
    )

    pair_symbol_date_counts = pair_ok_df.groupby("symbol")["date_yyyymmdd"].nunique()
    pair_symbols_both_dates = sorted(
        pair_symbol_date_counts.loc[pair_symbol_date_counts == 2].index.tolist()
    )
    both_dates_price_mask = panel_df["symbol"].isin(pair_symbols_both_dates)
    n_pair_symbols_with_price = int(panel_df.loc[both_dates_price_mask, PRICE_COL].notna().sum())
    n_pair_symbols_missing_price = int(panel_df.loc[both_dates_price_mask, PRICE_COL].isna().sum())

    input_overview_df = pd.DataFrame(
        [
            {
                "input_csv_path": str(resolved_input_csv_path),
                "cta_symbol_dir": str(resolved_cta_symbol_dir),
                "output_dir": str(resolved_output_dir),
                "input_csv_stem": input_csv_stem,
                "pair_label": f"{trade_exchange}{quote_exchange}",
                "n_pair_ok_rows": len(pair_ok_df),
                "n_pair_symbols_any_date": int(pair_ok_df["symbol"].nunique()),
                "n_pair_symbols_both_dates": len(pair_symbols_both_dates),
                "n_panel_symbols": int(panel_df["symbol"].nunique()),
                "n_panel_symbols_with_price": int(panel_df[PRICE_COL].notna().sum()),
                "n_both_dates_symbols_with_price": n_pair_symbols_with_price,
                "n_both_dates_symbols_missing_price": n_pair_symbols_missing_price,
            }
        ]
    )
    cta_price_summary_df = pd.DataFrame(
        [
            {
                "cta_symbol_dir": str(resolved_cta_symbol_dir),
                "price_column": price_column,
                "n_cta_files": len(cta_price_files),
                "n_cta_symbols": int(cta_mean_close_df["symbol"].nunique()),
                "n_pair_symbols_both_dates": len(pair_symbols_both_dates),
                "n_both_dates_symbols_with_price": n_pair_symbols_with_price,
                "n_both_dates_symbols_missing_price": n_pair_symbols_missing_price,
            }
        ]
    )

    outcome_specs_by_name = {
        str(spec["outcome"]): {
            "display_name": str(spec["display_name"]),
            "side": str(spec["side"]),
            "family": str(spec["family"]),
            "transform": str(spec["transform"]),
        }
        for spec in outcome_specs
    }
    available_y_stems = [str(spec["outcome"]) for spec in outcome_specs]
    outcome_list_df = outcome_specs_df.copy()
    return (
        available_y_stems,
        cta_mean_close_df,
        cta_price_summary_df,
        input_overview_df,
        outcome_list_df,
        outcome_specs_by_name,
        pair_ok_df,
        pair_symbols_both_dates,
        panel_df,
        round_lot_panel_df,
    )


@app.cell
def _(cta_price_summary_df):
    cta_price_summary_df
    return


@app.cell
def _(component_definition_df, mo):
    outcome_construction_section = mo.md(
        """
        ## Outcome Construction

        This section maps the six raw cross-K windows into side totals and
        within-side distance-band shares. It also carries the direct
        `closest_mode_to_zero_sec -> mode_position_sec` outcome, so the RD
        targets include 8 derived outcomes plus 1 direct mode-location outcome.
        """
    )
    outcome_construction_section
    component_definition_df
    return


@app.cell
def _(
    DATE_POST,
    DATE_PRE,
    DENSITY_REASON,
    DENSITY_STATUS,
    MIN_ANALYSIS_OBS,
    PRICE_COL,
    Path,
    UTC,
    analysis_note,
    build_rd_sample,
    datetime,
    default_balance_covariates,
    make_manual_rd_plot,
    pd,
    run_balance_tests,
    summarize_rdrobust,
    try_rdrobust,
    write_json,
):
    def run_outcome_workflow(
        raw_df,
        *,
        y_stem,
        cutoff,
        upper_bound,
        enforce_rule_consistency,
        bins_per_side,
        output_dir,
        balance_covariates,
        export_analysis_sample,
        export_balance_tests,
        outcome_specs_by_name,
        pre_round_lot_required,
    ):
        """Run the full RD workflow for one outcome and save artifacts.

        Parameters
        ----------
        raw_df
            Raw panel dataframe.
        y_stem
            Outcome stem.
        cutoff
            RD cutoff in dollars.
        upper_bound
            Upper support restriction for September mean close.
        enforce_rule_consistency
            Whether to enforce the post-reform round-lot rule.
        bins_per_side
            Plotting bins per side.
        output_dir
            Artifact directory.
        balance_covariates
            Optional fixed covariate list.
        export_analysis_sample
            Whether to save the analysis sample CSV.
        export_balance_tests
            Whether to save the balance test CSV.
        outcome_specs_by_name
            Outcome metadata keyed by outcome name.
        pre_round_lot_required
            Required pre-reform round lot.

        Returns
        -------
        tuple[dict[str, object], list[dict[str, object]], pathlib.Path | None]
            Summary dictionary, artifact records, and preview plot path.

        """
        outcome_spec = outcome_specs_by_name[y_stem]
        summary = {
            "status": "ok",
            "outcome": y_stem,
            "display_name": outcome_spec["display_name"],
            "side": outcome_spec["side"],
            "family": outcome_spec["family"],
            "transform": outcome_spec["transform"],
            "cutoff": cutoff,
            "upper_bound": upper_bound,
            "enforce_rule_consistency": enforce_rule_consistency,
            "density_status": DENSITY_STATUS,
            "density_reason": DENSITY_REASON,
            "n_total_raw": len(raw_df),
            "analysis_sample_csv": None,
            "balance_tests_csv": None,
            "plot_png": None,
            "summary_json": None,
        }
        artifact_rows = []
        preview_plot_path = None
        summary_path = output_dir / f"summary__{y_stem}.json"

        try:
            analysis_df = build_rd_sample(
                raw_df,
                y_stem=y_stem,
                cutoff=cutoff,
                upper_bound=upper_bound,
                enforce_rule_consistency=enforce_rule_consistency,
                outcome_specs_by_name=outcome_specs_by_name,
                pre_round_lot_required=pre_round_lot_required,
            )
            if len(analysis_df) < MIN_ANALYSIS_OBS:
                message = f"Too few observations after restrictions: n={len(analysis_df)}"
                raise ValueError(message)

            summary["n_analysis"] = len(analysis_df)
            summary["n_left"] = int((analysis_df["running"] <= 0).sum())
            summary["n_right"] = int((analysis_df["running"] > 0).sum())
            summary["n_actual_treated"] = int(analysis_df["D_actual"].sum())
            summary["n_rule_consistent"] = int(analysis_df["rule_consistent"].sum())

            if export_analysis_sample:
                sample_path = output_dir / f"analysis_sample__{y_stem}.csv"
                analysis_df.to_csv(sample_path, index=False)
                summary["analysis_sample_csv"] = str(sample_path)
                artifact_rows.append(
                    {
                        "outcome": y_stem,
                        "artifact_type": "analysis_sample_csv",
                        "path": str(sample_path),
                    }
                )

            covariates = (
                default_balance_covariates(raw_df, y_stem=y_stem)
                if balance_covariates is None
                else list(balance_covariates)
            )

            x = analysis_df[PRICE_COL].to_numpy(dtype=float)
            y = analysis_df["delta_y"].to_numpy(dtype=float)
            _bw, est, rd_meta = try_rdrobust(
                y=y,
                x=x,
                cutoff=cutoff,
            )
            rd_summary = summarize_rdrobust(est)
            rd_summary["bandwidth_source"] = rd_meta["bandwidth_source"]
            rd_summary["fallback_info"] = rd_meta["fallback_info"]
            rd_summary["auto_error"] = rd_meta["auto_error"]
            summary["rdrobust"] = rd_summary

            plot_path = output_dir / f"rd_plot__{y_stem}.png"
            make_manual_rd_plot(
                analysis_df,
                cutoff=cutoff,
                outpath=plot_path,
                y_label=f"Delta {outcome_spec['display_name']} ({outcome_spec['transform']})",
                title=f"RD at $250 cutoff: {outcome_spec['display_name']}",
                bins_per_side=bins_per_side,
                h_left=rd_summary["bandwidth_h_left"],
                h_right=rd_summary["bandwidth_h_right"],
            )
            summary["plot_png"] = str(plot_path)
            preview_plot_path = plot_path
            artifact_rows.append(
                {
                    "outcome": y_stem,
                    "artifact_type": "plot_png",
                    "path": str(plot_path),
                }
            )

            balance_df = run_balance_tests(
                analysis_df,
                covariates=covariates,
                cutoff=cutoff,
            )
            summary["n_balance_tests"] = len(balance_df)
            if export_balance_tests:
                balance_path = output_dir / f"balance__{y_stem}.csv"
                balance_df.to_csv(balance_path, index=False)
                summary["balance_tests_csv"] = str(balance_path)
                artifact_rows.append(
                    {
                        "outcome": y_stem,
                        "artifact_type": "balance_csv",
                        "path": str(balance_path),
                    }
                )

        except Exception as exc:  # noqa: BLE001
            summary["status"] = "error"
            summary["error_type"] = type(exc).__name__
            summary["error_message"] = str(exc)

        summary["summary_json"] = str(summary_path)
        write_json(summary, summary_path)
        artifact_rows.append(
            {
                "outcome": y_stem,
                "artifact_type": "summary_json",
                "path": str(summary_path),
            }
        )
        return summary, artifact_rows, preview_plot_path

    def run_batch_workflow(
        raw_df,
        *,
        available_y_stems,
        cutoff,
        upper_bound,
        enforce_rule_consistency,
        bins_per_side,
        output_dir,
        balance_covariates,
        export_analysis_sample,
        export_balance_tests,
        input_csv_path,
        cta_symbol_dir,
        price_column,
        pair_label,
        trade_exchange,
        quote_exchange,
        pre_round_lot_required,
        n_pair_ok_rows,
        n_pair_symbols_any_date,
        n_pair_symbols_both_dates,
        n_panel_symbols,
        n_panel_symbols_with_price,
        n_both_dates_symbols_with_price,
        n_both_dates_symbols_missing_price,
        n_round_lot_symbols,
        n_cta_files,
        n_cta_symbols,
        outcome_specs_by_name,
    ):
        """Run the RD export workflow for every available outcome.

        Parameters
        ----------
        raw_df
            Raw panel dataframe.
        available_y_stems
            Outcome stems to process.
        cutoff
            RD cutoff in dollars.
        upper_bound
            Upper support restriction for September mean close.
        enforce_rule_consistency
            Whether to enforce the post-reform round-lot rule.
        bins_per_side
            Plotting bins per side.
        output_dir
            Artifact directory.
        balance_covariates
            Optional fixed balance covariate list.
        export_analysis_sample
            Whether to save analysis sample CSVs.
        export_balance_tests
            Whether to save balance CSVs.
        input_csv_path
            Resolved cross-K summary path.
        cta_symbol_dir
            Resolved CTA price directory.
        price_column
            CTA price column used to build the running variable.
        pair_label
            Exchange-pair label.
        trade_exchange
            Trade exchange code.
        quote_exchange
            Quote exchange code.
        pre_round_lot_required
            Required pre-reform round lot.
        n_pair_ok_rows
            Count of ok pair rows in the long cross-K input.
        n_pair_symbols_any_date
            Count of pair symbols on either date.
        n_pair_symbols_both_dates
            Count of pair symbols observed on both dates.
        n_panel_symbols
            Count of symbols in the wide outcome panel.
        n_panel_symbols_with_price
            Count of symbols with any matched CTA mean close.
        n_both_dates_symbols_with_price
            Count of two-date symbols with matched CTA mean close.
        n_both_dates_symbols_missing_price
            Count of two-date symbols missing CTA mean close.
        n_round_lot_symbols
            Count of symbols with any round-lot data.
        n_cta_files
            Number of CTA September files read.
        n_cta_symbols
            Number of symbols in the CTA mean-close table.
        outcome_specs_by_name
            Outcome metadata keyed by outcome name.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, dict[str, object], pathlib.Path | None]
            Aggregate summary, artifact manifest, run manifest, and preview plot path.

        """
        summary_rows = []
        artifact_rows = []
        preview_plot_path = None

        for y_stem in available_y_stems:
            outcome_summary, outcome_artifacts, outcome_plot_path = run_outcome_workflow(
                raw_df,
                y_stem=y_stem,
                cutoff=cutoff,
                upper_bound=upper_bound,
                enforce_rule_consistency=enforce_rule_consistency,
                bins_per_side=bins_per_side,
                output_dir=output_dir,
                balance_covariates=balance_covariates,
                export_analysis_sample=export_analysis_sample,
                export_balance_tests=export_balance_tests,
                outcome_specs_by_name=outcome_specs_by_name,
                pre_round_lot_required=pre_round_lot_required,
            )
            summary_rows.append(outcome_summary)
            artifact_rows.extend(outcome_artifacts)
            if preview_plot_path is None and outcome_plot_path is not None:
                preview_plot_path = outcome_plot_path

        summary_all_df = pd.DataFrame(summary_rows)
        if not summary_all_df.empty and "rdrobust" in summary_all_df.columns:
            rd_fields = [
                "coef_robust",
                "p_robust",
                "bandwidth_source",
                "bandwidth_h_left",
                "bandwidth_h_right",
            ]
            for field_name in rd_fields:
                summary_all_df[field_name] = summary_all_df["rdrobust"].map(
                    lambda rd_summary, field_name=field_name: (
                        None if not isinstance(rd_summary, dict) else rd_summary.get(field_name)
                    )
                )
            summary_all_df["ci_robust_low"] = summary_all_df["rdrobust"].map(
                lambda rd_summary: (
                    None
                    if not isinstance(rd_summary, dict) or rd_summary.get("ci_robust") is None
                    else rd_summary["ci_robust"][0]
                )
            )
            summary_all_df["ci_robust_high"] = summary_all_df["rdrobust"].map(
                lambda rd_summary: (
                    None
                    if not isinstance(rd_summary, dict) or rd_summary.get("ci_robust") is None
                    else rd_summary["ci_robust"][1]
                )
            )
            summary_all_df = summary_all_df.drop(columns=["rdrobust"], errors="ignore")

        summary_all_path = output_dir / "summary_all.csv"
        summary_all_df.to_csv(summary_all_path, index=False)
        artifact_rows.append(
            {
                "outcome": "ALL",
                "artifact_type": "summary_all_csv",
                "path": str(summary_all_path),
            }
        )

        summary_all_json_path = output_dir / "summary_all.json"
        write_json({"rows": summary_rows}, summary_all_json_path)
        artifact_rows.append(
            {
                "outcome": "ALL",
                "artifact_type": "summary_all_json",
                "path": str(summary_all_json_path),
            }
        )

        artifact_manifest_df = pd.DataFrame(artifact_rows).sort_values(
            ["outcome", "artifact_type", "path"],
            kind="stable",
        )
        artifact_manifest_df["exists"] = artifact_manifest_df["path"].map(
            lambda path: Path(path).exists()
        )

        run_manifest = {
            "created_at_utc": datetime.now(UTC),
            "input_csv_path": str(input_csv_path),
            "cta_symbol_dir": str(cta_symbol_dir),
            "output_dir": str(output_dir),
            "pair_label": pair_label,
            "trade_exchange": trade_exchange,
            "quote_exchange": quote_exchange,
            "date_pre": DATE_PRE,
            "date_post": DATE_POST,
            "price_column": price_column,
            "running_variable": PRICE_COL,
            "cutoff": cutoff,
            "upper_bound": upper_bound,
            "pre_round_lot_required": pre_round_lot_required,
            "enforce_rule_consistency": enforce_rule_consistency,
            "available_y_stems": list(available_y_stems),
            "n_pair_ok_rows": n_pair_ok_rows,
            "n_pair_symbols_any_date": n_pair_symbols_any_date,
            "n_pair_symbols_both_dates": n_pair_symbols_both_dates,
            "n_panel_symbols": n_panel_symbols,
            "n_panel_symbols_with_price": n_panel_symbols_with_price,
            "n_both_dates_symbols_with_price": n_both_dates_symbols_with_price,
            "n_both_dates_symbols_missing_price": n_both_dates_symbols_missing_price,
            "n_round_lot_symbols": n_round_lot_symbols,
            "n_cta_files": n_cta_files,
            "n_cta_symbols": n_cta_symbols,
            "density_status": DENSITY_STATUS,
            "density_reason": DENSITY_REASON,
            "analysis_note": analysis_note,
            "artifact_files": artifact_manifest_df.to_dict(orient="records"),
        }
        run_manifest_path = output_dir / "run_manifest.json"
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

        return summary_all_df, artifact_manifest_df, run_manifest, preview_plot_path

    return (run_batch_workflow,)


@app.cell
def _(
    available_y_stems,
    balance_covariates,
    bins_per_side,
    cta_mean_close_df,
    cutoff,
    enforce_rule_consistency,
    export_analysis_sample,
    export_balance_tests,
    input_overview_df,
    outcome_specs_by_name,
    pair_label,
    pair_ok_df,
    pair_symbols_both_dates,
    panel_df,
    pre_round_lot_required,
    price_column,
    quote_exchange,
    resolved_cta_symbol_dir,
    resolved_input_csv_path,
    resolved_output_dir,
    round_lot_panel_df,
    run_batch_workflow,
    trade_exchange,
    upper_bound,
):
    input_row = input_overview_df.iloc[0]
    summary_all_df, artifact_manifest_df, run_manifest, preview_plot_path = run_batch_workflow(
        panel_df,
        available_y_stems=available_y_stems,
        cutoff=cutoff,
        upper_bound=upper_bound,
        enforce_rule_consistency=enforce_rule_consistency,
        bins_per_side=bins_per_side,
        output_dir=resolved_output_dir,
        balance_covariates=balance_covariates,
        export_analysis_sample=export_analysis_sample,
        export_balance_tests=export_balance_tests,
        input_csv_path=resolved_input_csv_path,
        cta_symbol_dir=resolved_cta_symbol_dir,
        price_column=price_column,
        pair_label=pair_label,
        trade_exchange=trade_exchange,
        quote_exchange=quote_exchange,
        pre_round_lot_required=pre_round_lot_required,
        n_pair_ok_rows=len(pair_ok_df),
        n_pair_symbols_any_date=int(pair_ok_df["symbol"].nunique()),
        n_pair_symbols_both_dates=len(pair_symbols_both_dates),
        n_panel_symbols=int(panel_df["symbol"].nunique()),
        n_panel_symbols_with_price=int(panel_df["mean_close_202509"].notna().sum()),
        n_both_dates_symbols_with_price=int(input_row["n_both_dates_symbols_with_price"]),
        n_both_dates_symbols_missing_price=int(input_row["n_both_dates_symbols_missing_price"]),
        n_round_lot_symbols=int(round_lot_panel_df["symbol"].nunique()),
        n_cta_files=len(sorted(resolved_cta_symbol_dir.glob("CTA.Symbol.File.202509*.csv"))),
        n_cta_symbols=int(cta_mean_close_df["symbol"].nunique()),
        outcome_specs_by_name=outcome_specs_by_name,
    )
    return (
        artifact_manifest_df,
        preview_plot_path,
        run_manifest,
        summary_all_df,
    )


@app.cell
def _(mo):
    results_section = mo.md(
        """
        ## Results

        This section reports input coverage and the per-outcome RD summaries.
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
def _(summary_all_df):
    summary_all_df
    return


@app.cell
def _(mo):
    artifacts_section = mo.md(
        """
        ## Artifacts

        This section lists the written files and shows one preview RD plot from
        the first successful outcome.
        """
    )
    artifacts_section
    return


@app.cell
def _(mo, mpimg, plt, preview_plot_path):
    if preview_plot_path is None:
        preview_display = mo.md("No successful RD plot was generated.")
    else:
        preview_plot_image = mpimg.imread(preview_plot_path)
        preview_plot_figure, axis = plt.subplots(figsize=(9, 5.2))
        axis.imshow(preview_plot_image)
        axis.axis("off")
        axis.set_title(str(preview_plot_path))
        preview_plot_figure.tight_layout()
        preview_display = preview_plot_figure
    preview_display
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
