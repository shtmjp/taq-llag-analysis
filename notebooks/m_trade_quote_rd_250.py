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

    return UTC, Path, datetime, json, mpimg, np, pd, plt, sm


@app.cell
def _(Path):
    input_csv_path = None
    output_base_dir = Path("data/derived/m_trade_quote_rd_250")
    cutoff = 250.0
    upper_bound = 1000.0
    enforce_rule_consistency = True
    bins_per_side = 10
    export_analysis_sample = True
    export_balance_tests = True
    balance_covariates = None
    return (
        balance_covariates,
        bins_per_side,
        cutoff,
        enforce_rule_consistency,
        export_analysis_sample,
        export_balance_tests,
        input_csv_path,
        output_base_dir,
        upper_bound,
    )


@app.cell
def _(Path, input_csv_path, output_base_dir):
    panel_base_dir = Path("data/derived/m_trade_quote_mode_symbol_panel")
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
    PRE_REFORM_ROUND_LOT = 100.0
    POST_REFORM_SMALL_LOT = 40.0
    DENSITY_STATUS = "omitted"
    DENSITY_REASON = "rddensity is incompatible with pandas 3 in the current environment."
    MIN_SIDE_OBS = 2
    MIN_BIN_EDGES = 2
    MIN_BALANCE_OBS = 10
    MIN_ANALYSIS_OBS = 20

    def infer_available_y_stems(columns):
        """Infer outcome stems that appear for both analysis dates.

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
        """Map an outcome stem to its delta transform.

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
        """Return the pre/post column names for one outcome stem.

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

        Returns
        -------
        pandas.DataFrame
            Analysis sample with derived RD columns.

        """
        pre_y_col, post_y_col = get_outcome_columns(y_stem)
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

        transform = transform_for_outcome(y_stem)
        analysis_df = df.copy()
        analysis_df = analysis_df.loc[analysis_df[PRICE_COL].notna()].copy()
        analysis_df = analysis_df.loc[analysis_df[PRICE_COL] < upper_bound].copy()
        analysis_df = analysis_df.loc[analysis_df[PRE_RL_COL] == PRE_REFORM_ROUND_LOT].copy()
        analysis_df = analysis_df.loc[analysis_df[POST_RL_COL].notna()].copy()

        analysis_df["y_pre"] = analysis_df[pre_y_col]
        analysis_df["y_post"] = analysis_df[post_y_col]
        analysis_df["delta_y"] = make_delta(
            analysis_df["y_pre"],
            analysis_df["y_post"],
            transform=transform,
        )
        analysis_df = analysis_df.loc[analysis_df["delta_y"].notna()].copy()

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
        if h_left <= 0 or h_right <= 0:
            message = "Bandwidths must be strictly positive."
            raise ValueError(message)
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

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(left_bins["x"], left_bins["y"], label="Left bins", color="#1d4ed8")
        ax.scatter(right_bins["x"], right_bins["y"], label="Right bins", color="#b91c1c")
        ax.plot(x_left, y_left, linewidth=2, color="#1d4ed8")
        ax.plot(x_right, y_right, linewidth=2, color="#b91c1c")
        ax.axvline(cutoff, linestyle="--", linewidth=1, color="#111827")
        ax.set_title(title)
        ax.set_xlabel("September mean close")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
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
        pre_suffix = f"_{DATE_PRE}"
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
        DATE_PRE,
        DENSITY_REASON,
        DENSITY_STATUS,
        MIN_ANALYSIS_OBS,
        PRICE_COL,
        build_rd_sample,
        default_balance_covariates,
        infer_available_y_stems,
        make_manual_rd_plot,
        run_balance_tests,
        summarize_rdrobust,
        transform_for_outcome,
        try_rdrobust,
        write_json,
    )


@app.cell
def _(  # noqa: C901, PLR0915
    DATE_PRE,
    MIN_ANALYSIS_OBS,
    DENSITY_REASON,
    DENSITY_STATUS,
    Path,
    UTC,
    datetime,
    PRICE_COL,
    build_rd_sample,
    default_balance_covariates,
    make_manual_rd_plot,
    pd,
    run_balance_tests,
    summarize_rdrobust,
    transform_for_outcome,
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

        Returns
        -------
        tuple[dict[str, object], list[dict[str, object]], pathlib.Path | None]
            Summary dictionary, artifact records, and preview plot path.

        """
        transform = transform_for_outcome(y_stem)
        summary = {
            "status": "ok",
            "outcome": y_stem,
            "transform": transform,
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
            )
            if len(analysis_df) < MIN_ANALYSIS_OBS:
                message = f"Too few observations after restrictions: n={len(analysis_df)}"
                raise ValueError(message)  # noqa: TRY301

            summary["n_analysis"] = len(analysis_df)
            summary["n_left"] = int((analysis_df["running"] <= 0).sum())
            summary["n_right"] = int((analysis_df["running"] > 0).sum())
            summary["n_actual_treated"] = int(analysis_df["D_actual"].sum())

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

            h_left = rd_summary["bandwidth_h_left"]
            h_right = rd_summary["bandwidth_h_right"]
            plot_path = output_dir / f"rd_plot__{y_stem}.png"
            make_manual_rd_plot(
                analysis_df,
                cutoff=cutoff,
                outpath=plot_path,
                y_label=f"Delta {y_stem} ({transform})",
                title=f"RD at $250 cutoff: {y_stem}",
                bins_per_side=bins_per_side,
                h_left=h_left,
                h_right=h_right,
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
            Resolved panel CSV path.

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
            "output_dir": str(output_dir),
            "date_pre": DATE_PRE,
            "cutoff": cutoff,
            "upper_bound": upper_bound,
            "enforce_rule_consistency": enforce_rule_consistency,
            "available_y_stems": list(available_y_stems),
            "density_status": DENSITY_STATUS,
            "density_reason": DENSITY_REASON,
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

    return run_batch_workflow, run_outcome_workflow


@app.cell
def _(infer_available_y_stems, pd, resolved_input_csv_path):
    raw_df = pd.read_csv(resolved_input_csv_path)
    available_y_stems = infer_available_y_stems(raw_df.columns)
    return available_y_stems, raw_df


@app.cell
def _(
    available_y_stems,
    balance_covariates,
    bins_per_side,
    cutoff,
    enforce_rule_consistency,
    export_analysis_sample,
    export_balance_tests,
    panel_csv_stem,
    pd,
    raw_df,
    resolved_input_csv_path,
    resolved_output_dir,
    run_batch_workflow,
    upper_bound,
):
    summary_all_df, artifact_manifest_df, run_manifest, preview_plot_path = run_batch_workflow(
        raw_df,
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
    )
    input_overview_df = pd.DataFrame(
        [
            {
                "input_csv_path": str(resolved_input_csv_path),
                "panel_csv_stem": panel_csv_stem,
                "output_dir": str(resolved_output_dir),
                "n_raw_rows": len(raw_df),
                "n_outcomes": len(available_y_stems),
            }
        ]
    )
    outcome_list_df = pd.DataFrame({"outcome": available_y_stems})
    return (
        artifact_manifest_df,
        input_overview_df,
        outcome_list_df,
        preview_plot_path,
        run_manifest,
        summary_all_df,
    )


@app.cell
def _(input_overview_df):
    input_overview_df  # noqa: B018
    return


@app.cell
def _(outcome_list_df):
    outcome_list_df  # noqa: B018
    return


@app.cell
def _(summary_all_df):
    summary_all_df  # noqa: B018
    return


@app.cell
def _(artifact_manifest_df):
    artifact_manifest_df  # noqa: B018
    return


@app.cell
def _(mpimg, plt, preview_plot_path):
    if preview_plot_path is None:
        preview_plot = None
    else:
        preview_image = mpimg.imread(preview_plot_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(preview_image)
        ax.axis("off")
        ax.set_title(str(preview_plot_path))
        fig.tight_layout()
        preview_plot = fig
    preview_plot  # noqa: B018
    return


@app.cell
def _(run_manifest):
    run_manifest  # noqa: B018
    return


if __name__ == "__main__":
    app.run()
