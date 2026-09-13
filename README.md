# Salmon Feed Trial — Data Flow Simulator

A Streamlit app that walks through, one stage at a time, how data moves
through a salmon feed trial station:

`Sensors + Technicians` → `Data Lake` → `Staging Area` → `Data Warehouse`
→ `Data Marts (per department)` → `Manager Dashboard`

Built on top of your original notebook's trial design (60-day low-protein
feed trial, 6 tanks / 3 treatment groups, EAV sensor + lab data, staging
validation, and star-schema warehouse) — refactored into reusable modules
and wired into a guided, stage-by-stage UI.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## How it works

- Each sidebar stage is revealed only after you complete the one before it —
  click "Next" at the bottom of a stage to advance.
- Data generated at Stage 1 (sensors/technicians) flows unchanged through every
  later stage, exactly like it would in a real pipeline — nothing is
  regenerated mid-run.
- **Restart simulation** (sidebar) resets everything and reruns Stage 1 with a
  new random seed, so you get a different — but structurally identical — trial
  each time.

## What changed vs. your notebook

- **Data marts were added.** Your notebook's final summary referenced three
  mart outputs (`ops_mart_daily_performance`, `research_mart_detailed`,
  `admin_mart_weekly_summary`) but never actually built them — the pipeline
  stopped at the warehouse fact table. `pipeline/marts.py` implements that
  missing layer:
  - **Ops Mart** (Farm Manager): daily per-tank water quality + alert counts.
  - **Research Mart** (Scientists): growth/stress detail plus ANOVA
    significance tests for treatment effects on weight gain and cortisol.
  - **Admin Mart** (Station Manager): weekly rollup across the whole trial.
- Functions now return `(result, log_lines)` instead of printing to stdout,
  so the app can render each stage's log as an expandable panel.
- `np.random.default_rng(seed)` replaces the global `np.random.seed(42)` so
  each "Restart simulation" produces an independent, reproducible run.

## Project layout

```
app.py                     Streamlit UI — the 7-stage walkthrough
pipeline/generation.py      Stage 1-2: trial metadata + sensor/technician data
pipeline/staging.py         Stage 3: cleaning & validation
pipeline/warehouse.py       Stage 4: star-schema dimension + fact tables
pipeline/marts.py           Stage 5: ops / research / admin data marts
```

## Known simplifications (worth knowing about)

- The fact table pivots sensor + lab readings to one row per tank+timestamp,
  which averages across the 10 sampled fish per tank. That leaves only 18
  tank-day data points for the significance tests (6 tanks × 3 sampling
  days) — enough to demonstrate the analysis, but too few to draw real
  biological conclusions from. A production version would keep fish-level
  rows in a separate fact table for the research mart.
- `fish_id` (F001–F010) is reused across tanks rather than being globally
  unique, carried over from the original notebook design — fine for this
  demo, but not how you'd model it in a real warehouse.
- Range-check thresholds rarely trigger with the current stress-event
  magnitude, so most runs show 0 flagged records — that reflects the
  simulated data, not a bug in the validation logic.
