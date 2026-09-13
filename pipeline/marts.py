"""
STAGE 5: DATA MARTS
--------------------
NOTE: this layer did not exist in the original notebook. The notebook's final
summary print claimed three mart files were produced (ops/research/admin),
but no code actually built them - the pipeline stopped at the warehouse
fact table. This module fills that gap so each department gets the view
suited to its own questions, all sourced from the same warehouse tables.

  - Operations Mart  -> Farm Manager:      daily water-quality + alerts, per tank
  - Research Mart     -> Scientists:        growth/stress detail + significance tests
  - Admin Mart        -> Station Manager:   trial-wide weekly rollup for reporting
"""

import warnings

import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

warnings.filterwarnings("ignore")


def _fact_with_tank_info(fact, dims):
    dim_tank = dims["dim_tank"][["tank_key", "tank_id", "treatment_group"]]
    merged = fact.merge(dim_tank, on="tank_key", how="left")
    merged["date"] = pd.to_datetime(merged["timestamp"]).dt.floor("D")
    return merged


def build_ops_mart(fact, dims, invalid_records):
    """Daily per-tank water-quality snapshot for the Farm Manager."""
    log = []
    merged = _fact_with_tank_info(fact, dims)

    sensor_cols = [c for c in ["water_temp_c", "do_pct", "ph_level", "salinity_ppt", "ammonia_mg_l"]
                   if c in merged.columns]
    daily = (
        merged.groupby(["tank_id", "treatment_group", "date"])[sensor_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    if len(invalid_records):
        inv = invalid_records.copy()
        inv["date"] = pd.to_datetime(inv["timestamp"]).dt.floor("D")
        alerts = inv.groupby(["tank_id", "date"]).size().rename("n_alerts").reset_index()
        daily = daily.merge(alerts, on=["tank_id", "date"], how="left")
        daily["n_alerts"] = daily["n_alerts"].fillna(0).astype(int)
    else:
        daily["n_alerts"] = 0

    daily = daily.sort_values(["tank_id", "date"]).reset_index(drop=True)
    log.append(f"ops_mart_daily_performance: {len(daily):,} tank-day rows, "
               f"{daily['n_alerts'].sum():,} total flagged alerts across the trial.")
    return daily, log


def build_research_mart(fact, dims):
    """Growth + stress-marker detail with treatment-effect significance tests,
    for scientists/statisticians."""
    log = []
    merged = _fact_with_tank_info(fact, dims)

    bio_cols = [c for c in ["weight_kg", "cortisol_ng_ml", "glucose_mg_dl"] if c in merged.columns]
    if not bio_cols:
        return pd.DataFrame(), {"note": "No lab measurements available yet."}, ["research_mart: no lab data yet."]

    detail = merged.dropna(subset=bio_cols, how="all").copy()
    detail = detail[["tank_id", "treatment_group", "date", "timestamp"] + bio_cols]

    # weight gain vs each tank's first recorded weight
    stats_summary = {}
    if "weight_kg" in detail.columns:
        baseline = (
            detail.dropna(subset=["weight_kg"])
            .sort_values("date")
            .groupby("tank_id")["weight_kg"].first()
            .rename("baseline_weight_kg")
        )
        detail = detail.merge(baseline, on="tank_id", how="left")
        detail["weight_gain_kg"] = detail["weight_kg"] - detail["baseline_weight_kg"]

    log.append(f"research_mart_detailed: {len(detail):,} tank-day records with lab measurements.")

    if _HAS_STATSMODELS:
        model_data = detail.dropna(subset=["weight_gain_kg"]) if "weight_gain_kg" in detail.columns else pd.DataFrame()
        if len(model_data) and model_data["treatment_group"].nunique() > 1:
            model = ols("weight_gain_kg ~ C(treatment_group)", data=model_data).fit()
            anova = sm.stats.anova_lm(model, typ=2)
            pval = float(anova.loc["C(treatment_group)", "PR(>F)"]) if "C(treatment_group)" in anova.index else None
            means = model_data.groupby("treatment_group")["weight_gain_kg"].mean().round(3).to_dict()
            stats_summary["weight_gain"] = {
                "p_value": pval,
                "significant": (pval is not None and pval < 0.05),
                "group_means_kg": means,
            }
            log.append(f"Weight-gain ANOVA by treatment: p={pval:.4f}" if pval is not None
                       else "Weight-gain ANOVA: could not compute p-value.")

        if "cortisol_ng_ml" in detail.columns:
            stress_data = detail.dropna(subset=["cortisol_ng_ml"])
            if len(stress_data) and stress_data["treatment_group"].nunique() > 1:
                stress_model = ols("cortisol_ng_ml ~ C(treatment_group)", data=stress_data).fit()
                stress_anova = sm.stats.anova_lm(stress_model, typ=2)
                pval = float(stress_anova.loc["C(treatment_group)", "PR(>F)"]) \
                    if "C(treatment_group)" in stress_anova.index else None
                stats_summary["cortisol"] = {
                    "p_value": pval,
                    "significant": (pval is not None and pval < 0.05),
                    "group_means": stress_data.groupby("treatment_group")["cortisol_ng_ml"].mean().round(2).to_dict(),
                }
                log.append(f"Cortisol ANOVA by treatment: p={pval:.4f}" if pval is not None
                           else "Cortisol ANOVA: could not compute p-value.")
    else:
        log.append("statsmodels not available - skipped significance testing.")

    return detail.sort_values(["tank_id", "date"]).reset_index(drop=True), stats_summary, log


def build_admin_mart(ops_mart, dims):
    """Trial-wide weekly rollup for the Station Manager / stakeholders."""
    log = []
    if ops_mart.empty:
        return pd.DataFrame(), ["admin_mart: no ops data available yet."]

    dim_date = dims["dim_date"][["date", "trial_week"]]
    merged = ops_mart.merge(dim_date, on="date", how="left")

    sensor_cols = [c for c in ["water_temp_c", "do_pct", "ph_level", "salinity_ppt", "ammonia_mg_l"]
                   if c in merged.columns]
    weekly = (
        merged.groupby(["trial_week", "treatment_group"])
        .agg({**{c: "mean" for c in sensor_cols}, "n_alerts": "sum"})
        .round(2)
        .reset_index()
        .sort_values(["trial_week", "treatment_group"])
    )
    log.append(f"admin_mart_weekly_summary: {len(weekly):,} week x treatment-group rows, "
               f"spanning trial weeks {int(weekly['trial_week'].min())}-{int(weekly['trial_week'].max())}.")
    return weekly, log
