"""
STAGE 1-2: DATA SOURCES + DATA LAKE
-----------------------------------
Simulates the two upstream sources at a feed trial station:
  - Automated sensors (water quality probes) writing high-frequency EAV
    (Entity-Attribute-Value) readings, as if dumped as CSVs into a data lake.
  - Technicians / lab staff taking manual measurements (fish weight, blood
    chemistry) on scheduled sampling days.

Adapted from the user's original notebook (dim_trial / dim_tank / dim_feed_batch
+ generate_sensor_data + generate_lab_data), refactored into reusable, seedable
functions for the Streamlit app.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TRIAL_START = datetime(2026, 1, 15, 0, 0, 0)


def get_trial_dimensions():
    """Static metadata describing the trial, its tanks, and its feed batches.

    This is the kind of reference data that would live in a trial-management
    system, not something sensors or technicians generate on the fly.
    """
    dim_trial = pd.DataFrame({
        "trial_id": ["T2026-01"],
        "trial_name": ["Low-Protein Feed Efficacy Trial"],
        "start_date": ["2026-01-15"],
        "end_date": ["2026-03-15"],
        "water_source": ["Fjord"],
        "tank_size_m3": [500],
        "salmon_strain": ["AquaGen QTL-5"],
        "starting_weight_avg_kg": [0.8],
        "trial_duration_days": [60],
    })

    dim_tank = pd.DataFrame({
        "tank_id": ["Tank_A1", "Tank_A2", "Tank_B1", "Tank_B2", "Tank_C1", "Tank_C2"],
        "trial_id": ["T2026-01"] * 6,
        "treatment_group": ["Control"] * 2 + ["NewFeed_Low"] * 2 + ["NewFeed_High"] * 2,
        "fish_count": [200, 200, 200, 200, 200, 200],
        "depth_m": [5, 5, 5, 5, 5, 5],
        "oxygen_system": ["Standard"] * 4 + ["Enhanced"] * 2,
    })

    dim_feed_batch = pd.DataFrame({
        "feed_batch_id": ["FB-C-01", "FB-C-02", "FB-NL-01", "FB-NL-02", "FB-NH-01", "FB-NH-02"],
        "feed_type": ["Control"] * 2 + ["NewFeed_Low"] * 2 + ["NewFeed_High"] * 2,
        "batch_date": ["2026-01-10", "2026-02-10", "2026-01-12", "2026-02-12", "2026-01-14", "2026-02-14"],
        "protein_pct": [42.0, 41.8, 38.5, 38.2, 35.0, 34.7],
        "lipid_pct": [18.0, 18.2, 22.0, 22.3, 25.0, 25.4],
        "carbohydrate_pct": [12.0, 12.0, 10.5, 10.5, 9.0, 9.0],
        "energy_mj_kg": [18.5, 18.6, 19.8, 19.9, 21.0, 21.1],
        "supplier": ["Skretting"] * 6,
    })

    return dim_trial, dim_tank, dim_feed_batch


def generate_sensor_data(dim_tank, days=60, readings_per_day=6, seed=None):
    """Simulate raw sensor readings in EAV format ("mimics CSV files landing
    in a data lake bucket, one row per parameter per reading")."""
    rng = np.random.default_rng(seed)
    records = []

    for tank in dim_tank["tank_id"]:
        base_temp = rng.normal(12.5, 0.5)
        base_oxygen = rng.normal(85, 3)
        base_ph = rng.normal(7.2, 0.1)
        base_salinity = rng.normal(32, 1)

        treatment = dim_tank.loc[dim_tank["tank_id"] == tank, "treatment_group"].iloc[0]
        feed_effect = rng.normal(1.5, 0.3) if "NewFeed" in treatment else 0

        for day in range(days):
            for reading in range(readings_per_day):
                timestamp = TRIAL_START + timedelta(days=day, hours=reading * 4)

                daily_cycle = 0.5 * np.sin(2 * np.pi * (reading / readings_per_day))
                temp = base_temp + daily_cycle + rng.normal(0, 0.3)
                oxygen = base_oxygen - 0.5 * daily_cycle + rng.normal(0, 1.5)

                # Simulated stress event mid-trial (e.g. a pump fault / warm spell)
                if 25 <= day <= 30:
                    temp += 2.0 + rng.normal(0, 0.5)
                    oxygen -= 8.0 + rng.normal(0, 2)

                if "NewFeed" in treatment:
                    oxygen -= feed_effect * 0.5

                parameters = [
                    ("water_temperature_c", temp, "°C"),
                    ("dissolved_oxygen_pct", oxygen, "%"),
                    ("ph_level", base_ph + rng.normal(0, 0.05), ""),
                    ("salinity_ppt", base_salinity + rng.normal(0, 0.2), "ppt"),
                    ("ammonia_mg_l", rng.exponential(0.5) + 0.1, "mg/L"),
                ]

                for param_name, value, unit in parameters:
                    records.append({
                        "tank_id": tank,
                        "trial_id": "T2026-01",
                        "timestamp": timestamp,
                        "parameter": param_name,
                        "value": round(float(value), 2),
                        "unit": unit,
                        "data_source": "sensor",
                    })

    return pd.DataFrame(records)


def generate_lab_data(dim_tank, measurement_days=(0, 30, 60), seed=None):
    """Simulate manual lab/technician measurements: fish weight + blood
    chemistry, sampled from 10 fish per tank on scheduled days."""
    rng = np.random.default_rng(None if seed is None else seed + 1)
    records = []

    for tank in dim_tank["tank_id"]:
        treatment = dim_tank.loc[dim_tank["tank_id"] == tank, "treatment_group"].iloc[0]

        if treatment == "Control":
            base_growth, growth_variation = 0.035, 0.008
        elif treatment == "NewFeed_Low":
            base_growth, growth_variation = 0.042, 0.009
        else:
            base_growth, growth_variation = 0.048, 0.010

        for day in measurement_days:
            for fish in range(10):
                weight = 0.8 + (day * base_growth) + rng.normal(0, growth_variation * 30)
                weight = max(0.5, weight)

                if treatment == "Control":
                    cortisol_base, glucose_base = 15, 80
                else:
                    cortisol_base, glucose_base = 12, 75

                if 25 <= day <= 30:
                    cortisol_base += 10
                    glucose_base += 15

                cortisol = cortisol_base + rng.normal(0, 3)
                glucose = glucose_base + rng.normal(0, 5)
                ts = datetime(2026, 1, 15, 8, 0, 0) + timedelta(days=day)
                fish_id = f"F{fish + 1:03d}"

                records.extend([
                    {"tank_id": tank, "trial_id": "T2026-01", "timestamp": ts,
                     "parameter": "weight_kg", "value": round(weight, 3), "unit": "kg",
                     "data_source": "manual_lab", "fish_id": fish_id},
                    {"tank_id": tank, "trial_id": "T2026-01", "timestamp": ts,
                     "parameter": "cortisol_ng_ml", "value": round(cortisol, 1), "unit": "ng/mL",
                     "data_source": "manual_lab", "fish_id": fish_id},
                    {"tank_id": tank, "trial_id": "T2026-01", "timestamp": ts,
                     "parameter": "glucose_mg_dl", "value": round(glucose, 1), "unit": "mg/dL",
                     "data_source": "manual_lab", "fish_id": fish_id},
                ])

    return pd.DataFrame(records)


def build_raw_data_lake(dim_tank, days=60, readings_per_day=6, seed=None):
    """STAGE 2: merge sensor stream + technician records into one raw data
    lake table -- untouched, uncleaned, exactly as each source produced it."""
    measurement_days = tuple(sorted({0, min(30, max(days - 1, 0)), max(days - 1, 0)}))
    sensor_data = generate_sensor_data(dim_tank, days=days, readings_per_day=readings_per_day, seed=seed)
    lab_data = generate_lab_data(dim_tank, measurement_days=measurement_days, seed=seed)
    raw_data_lake = pd.concat([sensor_data, lab_data], ignore_index=True)
    log = [
        f"Sensor stream: {len(sensor_data):,} EAV readings from {dim_tank['tank_id'].nunique()} tanks "
        f"({readings_per_day} readings/day x {days} days x 5 parameters).",
        f"Technician/lab records: {len(lab_data):,} manual observations "
        f"(10 fish x 3 parameters x {len(measurement_days)} sampling days x {dim_tank['tank_id'].nunique()} tanks).",
        f"Merged into raw_data_lake: {len(raw_data_lake):,} total rows, two data_source values "
        f"('sensor', 'manual_lab').",
    ]
    return raw_data_lake, sensor_data, lab_data, log
