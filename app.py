import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.generation import build_raw_data_lake, get_trial_dimensions
from pipeline.marts import build_admin_mart, build_ops_mart, build_research_mart
from pipeline.staging import stage_data_cleaning
from pipeline.warehouse import build_dimension_tables, build_fact_observations

st.set_page_config(page_title="Salmon Feed Trial — Data Flow Simulator", layout="wide")

STAGES = [
    ("Trial Setup", "🧪"),
    ("Data Sources", "📡"),
    ("Data Lake", "🌊"),
    ("Staging Area", "🧹"),
    ("Data Warehouse", "🏛️"),
    ("Data Marts", "🗂️"),
    ("Manager Dashboard", "📊"),
]


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state():
    defaults = {"stage_idx": 0, "max_stage": 0, "store": {}, "seed": 42}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def goto_next():
    st.session_state.stage_idx += 1
    st.session_state.max_stage = max(st.session_state.max_stage, st.session_state.stage_idx)


def restart(new_seed=True):
    st.session_state.stage_idx = 0
    st.session_state.max_stage = 0
    st.session_state.store = {}
    if new_seed:
        st.session_state.seed += 1


init_state()
store = st.session_state.store


def show_log(log_lines, label="Processing log"):
    with st.expander(f"🪵 {label}", expanded=False):
        for line in log_lines:
            st.markdown(f"- {line}")


def stage_header(idx):
    name, icon = STAGES[idx]
    st.markdown(f"## {icon} Stage {idx + 1} of {len(STAGES)}: {name}")


# ----------------------------------------------------------------------------
# Sidebar — pipeline stepper
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🐟 Feed Trial Data Flow")
    st.caption(
        "A simulation of how data moves through a salmon feed trial station: "
        "from sensors and technicians, through staging and the warehouse, "
        "out to each department, and finally to the station manager's dashboard."
    )
    visible = list(range(st.session_state.max_stage + 1))
    labels = []
    for i in visible:
        name, icon = STAGES[i]
        mark = "✅" if i < st.session_state.stage_idx else ("▶️" if i == st.session_state.stage_idx else "⬜")
        labels.append(f"{mark} {icon} {name}")
    picked = st.radio("Pipeline stage", options=visible, format_func=lambda i: labels[visible.index(i)],
                       index=st.session_state.stage_idx, label_visibility="collapsed")
    st.session_state.stage_idx = picked

    st.divider()
    if st.button("🔄 Restart simulation (new random trial run)", use_container_width=True):
        restart(new_seed=True)
        st.rerun()
    st.caption(f"Random seed: {st.session_state.seed}")

stage = st.session_state.stage_idx

# ============================================================================
# STAGE 0 — TRIAL SETUP
# ============================================================================
if stage == 0:
    stage_header(0)
    st.write(
        "Every trial station run starts from reference metadata that describes "
        "**what** is being tested, **where**, and **on what schedule** — this is set up "
        "once, before any sensor or technician data ever arrives."
    )

    dim_trial, dim_tank, dim_feed_batch = get_trial_dimensions()
    store["dim_trial"], store["dim_tank"], store["dim_feed_batch"] = dim_trial, dim_tank, dim_feed_batch

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Trial**")
        st.dataframe(dim_trial.T.rename(columns={0: "value"}), use_container_width=True)
    with c2:
        st.markdown("**Tanks & treatment groups**")
        st.dataframe(dim_tank, use_container_width=True, hide_index=True)

    st.markdown("**Feed batches**")
    st.dataframe(dim_feed_batch, use_container_width=True, hide_index=True)

    st.info(
        "6 tanks are split into 3 treatment groups (2 tanks each): **Control**, "
        "**NewFeed_Low** and **NewFeed_High** protein feed. The trial runs 60 days."
    )
    st.button("Start the trial → generate incoming data", type="primary", on_click=goto_next)

# ============================================================================
# STAGE 1 — DATA SOURCES
# ============================================================================
elif stage == 1:
    stage_header(1)
    st.write(
        "Two independent sources produce data at very different rates, in different "
        "formats, with no coordination between them — exactly how it works on a real station:"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📡 Automated sensors")
        st.write("Water-quality probes log readings every 4 hours, all day, every day: "
                 "temperature, dissolved oxygen, pH, salinity, ammonia.")
    with c2:
        st.markdown("#### 🧑‍🔬 Technicians / lab staff")
        st.write("On scheduled sampling days (day 0, 30, 60), technicians manually weigh "
                 "10 fish per tank and take blood samples for cortisol and glucose.")

    if "raw" not in store:
        with st.spinner("Simulating 60 days of sensor readings and lab sampling..."):
            raw, sensor, lab, log = build_raw_data_lake(store["dim_tank"], days=60, seed=st.session_state.seed)
        store["raw"], store["sensor"], store["lab"], store["log_sources"] = raw, sensor, lab, log

    m1, m2, m3 = st.columns(3)
    m1.metric("Sensor readings", f"{len(store['sensor']):,}")
    m2.metric("Technician records", f"{len(store['lab']):,}")
    m3.metric("Data sources", 2)

    t1, t2 = st.tabs(["Raw sensor stream (EAV)", "Raw technician records (EAV)"])
    with t1:
        st.dataframe(store["sensor"].head(20), use_container_width=True, hide_index=True)
    with t2:
        st.dataframe(store["lab"].head(20), use_container_width=True, hide_index=True)
    st.caption(
        "Both sources land in **EAV (Entity-Attribute-Value)** format — one row per "
        "parameter per reading — the way raw sensor/CSV data usually arrives before any modeling."
    )
    show_log(store["log_sources"])
    st.button("Send both streams to the data lake →", type="primary", on_click=goto_next)

# ============================================================================
# STAGE 2 — DATA LAKE
# ============================================================================
elif stage == 2:
    stage_header(2)
    st.write(
        "The two raw streams are simply **concatenated** — no cleaning, no validation yet. "
        "This is the data lake: a single landing zone holding everything exactly as it arrived."
    )
    raw = store["raw"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total raw rows", f"{len(raw):,}")
    m2.metric("From sensors", f"{(raw['data_source'] == 'sensor').sum():,}")
    m3.metric("From technicians", f"{(raw['data_source'] == 'manual_lab').sum():,}")

    fig = px.pie(raw, names="data_source", title="Raw data lake composition by source")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(raw.sample(min(15, len(raw)), random_state=1).sort_values("timestamp"),
                 use_container_width=True, hide_index=True)
    st.warning(
        "Nothing here has been checked yet — duplicate rows, missing values, sensor glitches "
        "and out-of-range readings can all be sitting in this table right now."
    )
    st.button("Send to staging for cleaning →", type="primary", on_click=goto_next)

# ============================================================================
# STAGE 3 — STAGING AREA
# ============================================================================
elif stage == 3:
    stage_header(3)
    st.write(
        "Before anything is trusted enough to load into the warehouse, it passes through "
        "**staging**: missing-value checks, type validation, de-duplication, and range checks "
        "against each parameter's expected limits."
    )

    if "staged" not in store:
        with st.spinner("Cleaning and validating raw records..."):
            staged, invalid, log = stage_data_cleaning(store["raw"])
        store["staged"], store["invalid"], store["log_staging"] = staged, invalid, log

    staged, invalid = store["staged"], store["invalid"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw rows in", f"{len(store['raw']):,}")
    m2.metric("Valid rows out", f"{len(staged):,}")
    m3.metric("Flagged for review", f"{len(invalid):,}", delta=None)

    if len(invalid):
        st.markdown("**Flagged out-of-range records**")
        st.dataframe(invalid[["tank_id", "parameter", "value", "quality_flag"]],
                     use_container_width=True, hide_index=True)
    else:
        st.success("No out-of-range readings this run — every record passed validation.")

    st.markdown("**Cleaned, audited sample**")
    st.dataframe(staged.head(10), use_container_width=True, hide_index=True)
    show_log(store["log_staging"])
    st.button("Load into the data warehouse →", type="primary", on_click=goto_next)

# ============================================================================
# STAGE 4 — DATA WAREHOUSE
# ============================================================================
elif stage == 4:
    stage_header(4)
    st.write(
        "Staged data is organized into a conventional **star schema**: descriptive "
        "*dimension* tables (date, tank, feed, fish, parameter) surrounding one wide "
        "*fact* table with one row per tank + timestamp and one column per parameter."
    )

    if "dims" not in store:
        with st.spinner("Building dimension tables and the fact table..."):
            dims, log3 = build_dimension_tables(store["staged"], store["dim_tank"], store["dim_feed_batch"])
            fact, log4 = build_fact_observations(store["staged"], dims)
        store["dims"], store["fact"], store["log_warehouse"] = dims, fact, log3 + log4

    dims, fact = store["dims"], store["fact"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fact rows", f"{len(fact):,}")
    m2.metric("Tanks", len(dims["dim_tank"]))
    m3.metric("Parameters", len(dims["dim_parameter"]))
    m4.metric("Trial days", len(dims["dim_date"]))

    tabs = st.tabs(["Fact table", "dim_date", "dim_tank", "dim_feed", "dim_parameter"])
    with tabs[0]:
        st.dataframe(fact.head(15), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(dims["dim_date"].head(10), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(dims["dim_tank"], use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(dims["dim_feed"], use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(dims["dim_parameter"], use_container_width=True, hide_index=True)

    show_log(store["log_warehouse"])
    st.button("Publish to department data marts →", type="primary", on_click=goto_next)

# ============================================================================
# STAGE 5 — DATA MARTS
# ============================================================================
elif stage == 5:
    stage_header(5)
    st.write(
        "The warehouse is built once, but each department needs a different **shape** of "
        "it. Data marts re-package the same warehouse data around each audience's questions."
    )

    if "ops_mart" not in store:
        with st.spinner("Building department data marts..."):
            ops, log5 = build_ops_mart(store["fact"], store["dims"], store["invalid"])
            research, stats_summary, log6 = build_research_mart(store["fact"], store["dims"])
            admin, log7 = build_admin_mart(ops, store["dims"])
        store["ops_mart"], store["research_mart"], store["stats_summary"], store["admin_mart"] = \
            ops, research, stats_summary, admin
        store["log_marts"] = log5 + log6 + log7

    tabs = st.tabs(["🚜 Ops Mart — Farm Manager", "🔬 Research Mart — Scientists",
                    "📋 Admin Mart — Station Manager"])
    with tabs[0]:
        st.caption("Daily per-tank water-quality snapshot, with alert counts.")
        st.dataframe(store["ops_mart"].head(15), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.caption("Growth + stress-marker detail, with treatment-effect significance tests.")
        st.dataframe(store["research_mart"], use_container_width=True, hide_index=True)
        stats = store["stats_summary"]
        if stats.get("weight_gain"):
            wg = stats["weight_gain"]
            verdict = "✅ significant" if wg["significant"] else "no significant difference"
            st.write(f"**Weight gain by treatment** — p = {wg['p_value']:.4f} ({verdict}). "
                     f"Group means (kg): {wg['group_means_kg']}")
        if stats.get("cortisol"):
            cs = stats["cortisol"]
            verdict = "✅ significant" if cs["significant"] else "no significant difference"
            st.write(f"**Cortisol by treatment** — p = {cs['p_value']:.4f} ({verdict}). "
                     f"Group means (ng/mL): {cs['group_means']}")
    with tabs[2]:
        st.caption("Trial-wide weekly rollup for reporting to stakeholders.")
        st.dataframe(store["admin_mart"], use_container_width=True, hide_index=True)

    show_log(store["log_marts"])
    st.button("Open the manager dashboard →", type="primary", on_click=goto_next)

# ============================================================================
# STAGE 6 — MANAGER DASHBOARD
# ============================================================================
elif stage == 6:
    stage_header(6)
    st.write(
        "The station manager doesn't query any of the tables above directly — they get "
        "a live dashboard that pulls from the marts."
    )

    ops, research, admin, stats = (
        store["ops_mart"], store["research_mart"], store["admin_mart"], store["stats_summary"]
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trial length", "60 days")
    m2.metric("Tanks monitored", ops["tank_id"].nunique())
    m3.metric("Total alerts logged", int(ops["n_alerts"].sum()))
    sig_count = sum(1 for v in stats.values() if v.get("significant"))
    m4.metric("Significant treatment effects", f"{sig_count} / {len(stats)}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(ops, x="date", y="water_temp_c", color="tank_id",
                      title="Water temperature by tank (note the mid-trial stress event)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(ops, x="date", y="do_pct", color="tank_id",
                      title="Dissolved oxygen by tank")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if not research.empty and "weight_gain_kg" in research.columns:
            fig = px.bar(research.groupby("treatment_group")["weight_gain_kg"].mean().reset_index(),
                         x="treatment_group", y="weight_gain_kg",
                         title="Average weight gain by treatment group")
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        alerts_by_group = ops.groupby("treatment_group")["n_alerts"].sum().reset_index()
        fig = px.bar(alerts_by_group, x="treatment_group", y="n_alerts",
                     title="Total water-quality alerts by treatment group")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Weekly summary (admin mart)")
    st.dataframe(admin, use_container_width=True, hide_index=True)

    st.success(
        "That's the full flow: sensors + technicians → data lake → staging → warehouse "
        "→ marts → dashboard. Restart from the sidebar to run it again with a new random trial."
    )
