"""
STAGE 4: DATA WAREHOUSE (star schema)
--------------------------------------
Turns the cleaned, staged EAV rows into a conventional star schema: a set of
dimension tables (date, tank, feed, fish, parameter) plus a wide fact table
with one row per tank+timestamp and one column per measured parameter.

Adapted from the user's notebook `build_dimension_tables` / `build_fact_observations`.
"""

import numpy as np
import pandas as pd


def build_dimension_tables(staged_data, dim_tank, dim_feed_batch):
    log = []

    # DIM_DATE
    all_dates = pd.date_range(
        start=staged_data["timestamp"].min().floor("D"),
        end=staged_data["timestamp"].max().ceil("D"),
        freq="D",
    )
    dim_date = pd.DataFrame({
        "date_key": range(1, len(all_dates) + 1),
        "date": all_dates,
        "year": all_dates.year,
        "month": all_dates.month,
        "day": all_dates.day,
        "day_of_week": all_dates.dayofweek,
        "weekday_name": all_dates.day_name(),
        "quarter": all_dates.quarter,
        "is_weekend": all_dates.dayofweek.isin([5, 6]),
        "trial_day": (all_dates - pd.Timestamp("2026-01-15")).days + 1,
    })
    dim_date["trial_week"] = ((dim_date["trial_day"] - 1) // 7) + 1
    log.append(f"DIM_DATE: {len(dim_date)} calendar-day records.")

    # DIM_TANK
    dim_tank_wh = dim_tank[["tank_id", "treatment_group", "fish_count", "depth_m", "oxygen_system"]].copy()
    dim_tank_wh["tank_key"] = range(1, len(dim_tank_wh) + 1)
    log.append(f"DIM_TANK: {len(dim_tank_wh)} tank records.")

    # DIM_FEED
    dim_feed = dim_feed_batch[["feed_batch_id", "feed_type", "protein_pct", "lipid_pct",
                               "carbohydrate_pct", "energy_mj_kg", "supplier"]].copy()
    dim_feed["feed_key"] = range(1, len(dim_feed) + 1)
    log.append(f"DIM_FEED: {len(dim_feed)} feed-batch records.")

    # DIM_FISH
    if "fish_id" in staged_data.columns:
        dim_fish = staged_data[["fish_id"]].drop_duplicates().dropna().copy()
        dim_fish["fish_key"] = range(1, len(dim_fish) + 1)
        log.append(f"DIM_FISH: {len(dim_fish)} sampled-fish records.")
    else:
        dim_fish = None
        log.append("DIM_FISH: skipped (no fish_id in staged data).")

    # DIM_PARAMETER
    dim_parameter = staged_data[["parameter", "unit"]].drop_duplicates().copy()
    dim_parameter["parameter_key"] = range(1, len(dim_parameter) + 1)
    log.append(f"DIM_PARAMETER: {len(dim_parameter)} distinct parameters.")

    dims = {
        "dim_date": dim_date,
        "dim_tank": dim_tank_wh,
        "dim_feed": dim_feed,
        "dim_fish": dim_fish,
        "dim_parameter": dim_parameter,
    }
    return dims, log


def build_fact_observations(staged_data, dimension_tables):
    log = []

    if "fish_id" in staged_data.columns:
        pivot = staged_data.pivot_table(
            index=["tank_id", "timestamp"], columns="parameter", values="value", aggfunc="mean"
        ).reset_index()
    else:
        pivot = staged_data.pivot_table(
            index=["tank_id", "timestamp"], columns="parameter", values="value", aggfunc="first"
        ).reset_index()
    pivot.columns = [str(c).strip() for c in pivot.columns]
    log.append(f"Pivoted EAV rows to wide format: {pivot.shape[0]:,} rows x {pivot.shape[1]} columns.")

    pivot["date"] = pd.to_datetime(pivot["timestamp"]).dt.floor("D")
    date_key_map = dimension_tables["dim_date"].set_index("date")["date_key"]
    pivot["date_key"] = pivot["date"].map(date_key_map)

    tank_key_map = dimension_tables["dim_tank"].set_index("tank_id")["tank_key"]
    pivot["tank_key"] = pivot["tank_id"].map(tank_key_map)

    pivot["feed_key"] = 1  # simplified: one active feed batch assumed per period, as in source notebook

    numeric_cols = pivot.select_dtypes(include=[np.number]).columns
    pivot[numeric_cols] = pivot[numeric_cols].round(4)

    param_cols = [c for c in pivot.columns if c not in
                  ["tank_id", "timestamp", "date", "date_key", "tank_key", "feed_key"]]
    fact = pivot[["date_key", "tank_key", "feed_key", "timestamp"] + param_cols].copy()
    fact["fact_key"] = range(1, len(fact) + 1)
    log.append(f"Fact table built: {fact.shape[0]:,} rows x {fact.shape[1]} columns "
               f"(one row per tank + timestamp).")

    return fact, log
