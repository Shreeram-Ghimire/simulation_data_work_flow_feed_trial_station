"""
STAGE 3: STAGING AREA
----------------------
Cleans and validates the merged raw data lake before it's trusted enough to
load into the warehouse: missing values, type coercion, duplicates, and
range checks against expected sensor/lab limits.

Adapted from the user's notebook `stage_data_cleaning`, refactored to return
structured results + a step-by-step log instead of printing to stdout.
"""

from datetime import datetime

import pandas as pd

PARAM_RANGES = {
    "water_temperature_c": (0, 25),
    "dissolved_oxygen_pct": (0, 100),
    "ph_level": (6.5, 8.5),
    "salinity_ppt": (25, 40),
    "ammonia_mg_l": (0, 5),
    "weight_kg": (0.3, 5.0),
    "cortisol_ng_ml": (0, 50),
    "glucose_mg_dl": (30, 200),
}

PARAMETER_RENAME = {
    "water_temperature_c": "water_temp_c",
    "dissolved_oxygen_pct": "do_pct",
}


def _check_range(row):
    lo_hi = PARAM_RANGES.get(row["parameter"])
    if lo_hi and not (lo_hi[0] <= row["value"] <= lo_hi[1]):
        return "out_of_range"
    return "valid"


def stage_data_cleaning(raw_data):
    """Run the staging pipeline. Returns (staged_data, invalid_records, log)."""
    log = []
    staged = raw_data.copy()
    start_rows = len(staged)

    # 1. Missing critical fields
    missing = staged.isnull().sum()
    critical_missing = staged[["tank_id", "timestamp", "parameter", "value"]].isnull().sum().sum()
    if critical_missing:
        before = len(staged)
        staged = staged.dropna(subset=["tank_id", "timestamp", "parameter", "value"])
        log.append(f"Missing-value check: dropped {before - len(staged):,} rows missing critical fields.")
    else:
        log.append("Missing-value check: no missing values in critical fields.")

    # 2. Type validation
    staged["timestamp"] = pd.to_datetime(staged["timestamp"])
    staged["value"] = pd.to_numeric(staged["value"], errors="coerce")
    before = len(staged)
    staged = staged.dropna(subset=["value"])
    dropped_nonnumeric = before - len(staged)
    log.append(f"Type validation: {dropped_nonnumeric:,} rows dropped for non-numeric values, "
               f"timestamps coerced to datetime.")

    # 3. Duplicates
    subset = ["tank_id", "timestamp", "parameter", "fish_id"] if "fish_id" in staged.columns \
        else ["tank_id", "timestamp", "parameter"]
    before = len(staged)
    staged = staged.drop_duplicates(subset=subset)
    log.append(f"Duplicate check: removed {before - len(staged):,} duplicate records "
               f"(matched on {', '.join(subset)}).")

    # 4. Range validation
    staged["quality_flag"] = staged.apply(_check_range, axis=1)
    invalid_records = staged[staged["quality_flag"] == "out_of_range"].copy()
    staged = staged[staged["quality_flag"] == "valid"].copy()
    if len(invalid_records):
        log.append(f"Range validation: flagged {len(invalid_records):,} out-of-range readings "
                   f"(sent to invalid_records for review, not loaded to the warehouse).")
    else:
        log.append("Range validation: all readings within expected ranges.")

    # 5. Standardize parameter names
    staged["parameter"] = staged["parameter"].replace(PARAMETER_RENAME)
    log.append("Standardized parameter names (e.g. water_temperature_c -> water_temp_c).")

    # 6. Audit columns
    staged["etl_processed_date"] = datetime.now()
    staged["etl_batch_id"] = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log.append("Added audit columns: etl_processed_date, etl_batch_id.")

    log.append(f"Staging complete: {start_rows:,} raw rows in -> {len(staged):,} valid rows out, "
               f"{len(invalid_records):,} flagged for review.")

    return staged, invalid_records, log
