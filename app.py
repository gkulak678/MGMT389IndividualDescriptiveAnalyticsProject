# ------------------------------------------------------------
# CRM Sales Opportunities Dashboard
# Individual Project Final Submission | Descriptive Analytics
# ------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="CRM Sales Opportunities Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Light CSS for cleaner executive look
# -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .section-box {
        border-left: 4px solid #d0d7de;
        background: #f8fafc;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .small-note {color: #6b7280; font-size: 0.86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Formatting helpers
# -----------------------------
def money(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:.{decimals}f}M"
    if x >= 1_000:
        return f"{sign}${x / 1_000:.{decimals}f}K"
    return f"{sign}${x:,.0f}"


def pct(x, decimals=1):
    if pd.isna(x):
        return "N/A"
    return f"{float(x) * 100:.{decimals}f}%"


def whole(x):
    if pd.isna(x):
        return "N/A"
    return f"{float(x):,.0f}"


def safe_divide(num, den):
    return np.where(den == 0, np.nan, num / den)


def table_format(df):
    format_dict = {}
    for col in df.columns:
        low = col.lower()
        if "rate" in low or "share" in low or "realization" in low:
            format_dict[col] = "{:.1%}"
        elif "revenue" in low or "value" in low or "deal" in low or "price" in low:
            if pd.api.types.is_numeric_dtype(df[col]):
                format_dict[col] = "${:,.0f}"
        elif "days" in low:
            if pd.api.types.is_numeric_dtype(df[col]):
                format_dict[col] = "{:.1f}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            format_dict[col] = "{:,.0f}"
    return df.style.format(format_dict, na_rep="—")


# -----------------------------
# Data loading
# -----------------------------
FILE_HINTS = {
    "pipeline": "sales_pipeline",
    "accounts": "accounts",
    "products": "products",
    "teams": "sales_teams",
    "dictionary": "data_dictionary",
}


def find_local_csv(keyword):
    """Looks for files like sales_pipeline.csv or sales_pipeline(1).csv."""
    search_folders = [Path.cwd(), Path.cwd() / "data", Path("/mnt/data")]
    for folder in search_folders:
        if folder.exists():
            matches = sorted(folder.glob(f"*{keyword}*.csv"))
            if matches:
                return matches[0]
    return None


@st.cache_data(show_spinner=False)
def read_local_csv(path_string):
    return pd.read_csv(path_string)


def load_csvs():
    """Loads files from the app folder first; sidebar upload is a backup option."""
    with st.sidebar.expander("Data source", expanded=False):
        st.write(
            "The app first looks for the CSVs in the app folder or a `data/` folder. "
            "If it cannot find them, upload the CSVs here."
        )
        uploaded_files = st.file_uploader(
            "Optional CSV upload",
            type="csv",
            accept_multiple_files=True,
        )

    loaded = {}
    loaded_notes = []
    missing = []

    for name, keyword in FILE_HINTS.items():
        local_path = find_local_csv(keyword)
        if local_path is not None:
            loaded[name] = read_local_csv(str(local_path))
            loaded_notes.append(f"{name}: `{local_path.name}`")
            continue

        matched_upload = None
        for file in uploaded_files or []:
            if keyword in file.name.lower():
                matched_upload = file
                break

        if matched_upload is not None:
            loaded[name] = pd.read_csv(matched_upload)
            loaded_notes.append(f"{name}: uploaded file")
        elif name != "dictionary":
            missing.append(keyword)

    if missing:
        st.error(
            "Missing required CSV file(s): "
            + ", ".join(missing)
            + ". Add them to the app folder, add them to a `data/` folder, or upload them in the sidebar."
        )
        st.stop()

    if "dictionary" not in loaded:
        loaded["dictionary"] = pd.DataFrame(columns=["Table", "Field", "Description"])

    return loaded, loaded_notes


# -----------------------------
# Data cleaning and joining
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare_data(pipeline_raw, accounts_raw, products_raw, teams_raw):
    pipeline = pipeline_raw.copy()
    accounts = accounts_raw.copy()
    products = products_raw.copy()
    teams = teams_raw.copy()

    # Clean column names.
    for frame in [pipeline, accounts, products, teams]:
        frame.columns = [str(c).strip() for c in frame.columns]

    # Trim text columns while preserving missing values.
    for frame in [pipeline, accounts, products, teams]:
        for col in frame.select_dtypes(include="object").columns:
            frame[col] = frame[col].apply(lambda x: str(x).strip() if pd.notna(x) else np.nan)

    # Known dataset corrections.
    pipeline["product_clean"] = pipeline["product"].replace({"GTXPro": "GTX Pro"})
    products["product_clean"] = products["product"].replace({"GTXPro": "GTX Pro"})
    accounts["sector_clean"] = accounts["sector"].replace({"technolgy": "technology"}).str.title()
    accounts["office_location_clean"] = accounts["office_location"].replace({"Philipines": "Philippines"})

    # Convert dates and numeric fields.
    pipeline["engage_date"] = pd.to_datetime(pipeline["engage_date"], errors="coerce")
    pipeline["close_date"] = pd.to_datetime(pipeline["close_date"], errors="coerce")
    pipeline["close_value"] = pd.to_numeric(pipeline["close_value"], errors="coerce")
    products["sales_price"] = pd.to_numeric(products["sales_price"], errors="coerce")
    accounts["revenue"] = pd.to_numeric(accounts["revenue"], errors="coerce")
    accounts["employees"] = pd.to_numeric(accounts["employees"], errors="coerce")
    accounts["year_established"] = pd.to_numeric(accounts["year_established"], errors="coerce")

    # Analytical fields.
    pipeline["is_closed"] = pipeline["deal_stage"].isin(["Won", "Lost"])
    pipeline["is_open"] = pipeline["deal_stage"].isin(["Engaging", "Prospecting"])
    pipeline["won_flag"] = np.where(
        pipeline["deal_stage"].eq("Won"),
        1,
        np.where(pipeline["deal_stage"].eq("Lost"), 0, np.nan),
    )
    pipeline["lost_flag"] = np.where(pipeline["deal_stage"].eq("Lost"), 1, 0)
    pipeline["won_value"] = np.where(
        pipeline["deal_stage"].eq("Won"),
        pipeline["close_value"].fillna(0),
        0,
    )
    pipeline["days_to_close"] = (pipeline["close_date"] - pipeline["engage_date"]).dt.days
    pipeline.loc[~pipeline["is_closed"], "days_to_close"] = np.nan
    pipeline["engage_quarter"] = pipeline["engage_date"].dt.to_period("Q").astype("string")
    pipeline["close_quarter"] = pipeline["close_date"].dt.to_period("Q").astype("string")
    pipeline["close_month"] = pipeline["close_date"].dt.to_period("M").astype("string")

    def employee_segment(x):
        if pd.isna(x):
            return "Unknown"
        if x < 500:
            return "Small (<500)"
        if x < 2_000:
            return "Mid-Market (500-1,999)"
        if x < 10_000:
            return "Large (2,000-9,999)"
        return "Enterprise (10,000+)"

    def revenue_segment(x):
        if pd.isna(x):
            return "Unknown"
        if x < 100:
            return "<$100M"
        if x < 500:
            return "$100M-$499M"
        if x < 1_000:
            return "$500M-$999M"
        return "$1B+"

    accounts["employee_segment"] = accounts["employees"].apply(employee_segment)
    accounts["revenue_segment"] = accounts["revenue"].apply(revenue_segment)
    accounts["company_age_at_close_period"] = 2017 - accounts["year_established"]

    products_join = products.drop(columns=["product"], errors="ignore")

    df = (
        pipeline.merge(teams, on="sales_agent", how="left")
        .merge(products_join, on="product_clean", how="left")
        .merge(accounts, on="account", how="left", suffixes=("", "_account"))
    )

    # Display-friendly fields.
    df["product"] = df["product_clean"]
    df["manager"] = df["manager"].fillna("Unassigned")
    df["regional_office"] = df["regional_office"].fillna("Unassigned")
    df["series"] = df["series"].fillna("Unknown")
    df["sector_display"] = df["sector_clean"].fillna("Unknown / Not Linked")
    df["office_location_display"] = df["office_location_clean"].fillna("Unknown / Not Linked")
    df["employee_segment"] = df["employee_segment"].fillna("Unknown")
    df["revenue_segment"] = df["revenue_segment"].fillna("Unknown")
    df["price_realization"] = np.where(
        df["sales_price"].fillna(0).eq(0),
        np.nan,
        df["close_value"] / df["sales_price"],
    )
    df["won_price_realization"] = np.where(df["deal_stage"].eq("Won"), df["price_realization"], np.nan)

    return df, accounts, products, teams


# -----------------------------
# KPI functions
# -----------------------------
def overall_kpis(data):
    closed = data[data["is_closed"]].copy()
    closed_count = closed["opportunity_id"].nunique()
    won_count = int(closed["deal_stage"].eq("Won").sum())
    lost_count = int(closed["deal_stage"].eq("Lost").sum())
    won_revenue = float(closed.loc[closed["deal_stage"].eq("Won"), "close_value"].sum())
    win_rate = won_count / closed_count if closed_count else np.nan
    avg_days = closed["days_to_close"].mean() if closed_count else np.nan
    avg_won_deal = won_revenue / won_count if won_count else np.nan

    return {
        "total_opps": data["opportunity_id"].nunique(),
        "closed_opps": closed_count,
        "open_opps": int(data["is_open"].sum()),
        "won_deals": won_count,
        "lost_deals": lost_count,
        "win_rate": win_rate,
        "won_revenue": won_revenue,
        "avg_days": avg_days,
        "avg_won_deal": avg_won_deal,
    }


def summarize_closed(data, group_cols, min_closed=0):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    closed = data[data["is_closed"]].copy()
    if closed.empty:
        return pd.DataFrame(columns=group_cols)

    out = (
        closed.groupby(group_cols, dropna=False)
        .agg(
            closed_opportunities=("opportunity_id", "nunique"),
            won_deals=("won_flag", "sum"),
            lost_deals=("lost_flag", "sum"),
            won_revenue=("won_value", "sum"),
            avg_close_value=("close_value", "mean"),
            avg_won_deal=("won_value", lambda s: s[s > 0].mean()),
            avg_days_to_close=("days_to_close", "mean"),
            avg_price_realization=("won_price_realization", "mean"),
        )
        .reset_index()
    )

    out["won_deals"] = out["won_deals"].astype(int)
    out["lost_deals"] = out["lost_deals"].astype(int)
    out["win_rate"] = safe_divide(out["won_deals"], out["closed_opportunities"])
    out["revenue_per_closed_opp"] = safe_divide(out["won_revenue"], out["closed_opportunities"])
    out["revenue_share"] = safe_divide(out["won_revenue"], out["won_revenue"].sum())

    if min_closed > 0:
        out = out[out["closed_opportunities"] >= min_closed]

    return out.sort_values("won_revenue", ascending=False)


def stage_mix(data, group_col):
    out = (
        data.groupby([group_col, "deal_stage"], dropna=False)["opportunity_id"]
        .nunique()
        .reset_index(name="opportunities")
    )
    return out


def add_agent_flag(agent_df):
    out = agent_df.copy()
    if out.empty:
        return out

    median_win = out["win_rate"].median(skipna=True)
    median_rev_eff = out["revenue_per_closed_opp"].median(skipna=True)
    median_days = out["avg_days_to_close"].median(skipna=True)

    conditions = [
        (out["win_rate"] < median_win) & (out["revenue_per_closed_opp"] < median_rev_eff),
        (out["win_rate"] < median_win) & (out["avg_days_to_close"] > median_days),
        (out["win_rate"] >= median_win) & (out["revenue_per_closed_opp"] >= median_rev_eff),
    ]
    choices = [
        "Review: low win rate + low revenue efficiency",
        "Review: low win rate + longer cycle",
        "Strong: above median win rate + revenue efficiency",
    ]
    out["diagnostic_flag"] = np.select(conditions, choices, default="Monitor / mixed performance")
    return out


def executive_summary_text(data):
    k = overall_kpis(data)
    product_summary = summarize_closed(data, "product")
    region_summary = summarize_closed(data, "regional_office")
    agent_summary = add_agent_flag(summarize_closed(data, ["sales_agent", "manager", "regional_office"], min_closed=15))

    lines = [
        f"The selected pipeline view includes {whole(k['total_opps'])} total opportunities, with {whole(k['closed_opps'])} closed opportunities and {whole(k['open_opps'])} open opportunities.",
        f"Closed deals converted at a {pct(k['win_rate'])} win rate and generated {money(k['won_revenue'])} in won revenue.",
    ]

    if not region_summary.empty:
        r = region_summary.iloc[0]
        lines.append(
            f"The top region by won revenue is {r['regional_office']}, generating {money(r['won_revenue'])} with a {pct(r['win_rate'])} win rate."
        )

    if not product_summary.empty:
        p = product_summary.iloc[0]
        lines.append(
            f"The top product by won revenue is {p['product']}, generating {money(p['won_revenue'])} with a {pct(p['win_rate'])} win rate."
        )

    if not agent_summary.empty:
        review_count = int(agent_summary["diagnostic_flag"].str.startswith("Review").sum())
        lines.append(
            f"The diagnostic view flags {review_count} agent(s) for potential coaching review based on relative conversion, revenue efficiency, and sales-cycle performance."
        )

    return " ".join(lines)


# -----------------------------
# App data setup
# -----------------------------
loaded, source_notes = load_csvs()
df, accounts_clean, products_clean, teams_clean = prepare_data(
    loaded["pipeline"], loaded["accounts"], loaded["products"], loaded["teams"]
)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.title("Dashboard Filters")
st.sidebar.caption("Filters apply across all tabs.")

with st.sidebar.expander("Files loaded", expanded=False):
    for note in source_notes:
        st.write(f"- {note}")

min_date = df["engage_date"].min().date()
max_date = df["engage_date"].max().date()
date_range = st.sidebar.date_input(
    "Engage date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="The main filter uses engage date so open and closed opportunities can be analyzed together.",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

regions = sorted(df["regional_office"].dropna().unique())
selected_regions = st.sidebar.multiselect("Region", regions, default=regions)

managers = sorted(df[df["regional_office"].isin(selected_regions)]["manager"].dropna().unique())
selected_managers = st.sidebar.multiselect("Manager", managers, default=managers)

agents = sorted(df[df["manager"].isin(selected_managers)]["sales_agent"].dropna().unique())
selected_agents = st.sidebar.multiselect("Sales Agent", agents, default=agents)

products = sorted(df["product"].dropna().unique())
selected_products = st.sidebar.multiselect("Product", products, default=products)

stages = ["Won", "Lost", "Engaging", "Prospecting"]
selected_stages = st.sidebar.multiselect("Deal Stage", stages, default=stages)

sectors = sorted(df["sector_display"].dropna().unique())
selected_sectors = st.sidebar.multiselect("Account Sector", sectors, default=sectors)

filtered = df[
    (df["engage_date"].dt.date >= start_date)
    & (df["engage_date"].dt.date <= end_date)
    & (df["regional_office"].isin(selected_regions))
    & (df["manager"].isin(selected_managers))
    & (df["sales_agent"].isin(selected_agents))
    & (df["product"].isin(selected_products))
    & (df["deal_stage"].isin(selected_stages))
    & (df["sector_display"].isin(selected_sectors))
].copy()

if filtered.empty:
    st.warning("No records match the current filters. Adjust the sidebar filters to continue.")
    st.stop()

kpis = overall_kpis(filtered)

# -----------------------------
# Header
# -----------------------------
st.title("CRM Sales Opportunities Dashboard")
st.caption("Descriptive analytics solution for evaluating B2B hardware sales pipeline performance.")

st.markdown(
    """
    <div class="section-box">
    <b>Business purpose:</b> This dashboard helps a VP of Sales, RevOps leader, or commercial finance stakeholder compare pipeline performance across teams, agents, products, quarters, and account segments. The goal is not to predict future sales, but to make historical CRM performance easier to diagnose and act on.
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
metric_cols[0].metric("Total Opportunities", whole(kpis["total_opps"]))
metric_cols[1].metric("Closed Opportunities", whole(kpis["closed_opps"]))
metric_cols[2].metric("Win Rate", pct(kpis["win_rate"]))
metric_cols[3].metric("Won Revenue", money(kpis["won_revenue"]))
metric_cols[4].metric("Avg Days to Close", f"{kpis['avg_days']:.1f}" if not pd.isna(kpis["avg_days"]) else "N/A")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_team, tab_agent, tab_product, tab_trends, tab_segments, tab_data = st.tabs(
    [
        "1. Executive Overview",
        "2. Team Performance",
        "3. Agent Diagnostics",
        "4. Product Analysis",
        "5. Quarterly Trends",
        "6. Account Segments",
        "7. Data Explorer",
    ]
)

# -----------------------------
# 1. Executive Overview
# -----------------------------
with tab_overview:
    st.subheader("Executive Overview")
    st.write("This page gives the high-level readout a non-technical decision maker would need first.")
    st.success(executive_summary_text(filtered))

    c1, c2 = st.columns(2)

    with c1:
        stage_summary = (
            filtered.groupby("deal_stage")["opportunity_id"]
            .nunique()
            .reindex(stages)
            .dropna()
            .reset_index(name="opportunities")
        )
        fig = px.bar(
            stage_summary,
            x="deal_stage",
            y="opportunities",
            text="opportunities",
            title="Pipeline Stage Mix",
            labels={"deal_stage": "Deal Stage", "opportunities": "Opportunities"},
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(height=420, xaxis_title="", yaxis_title="Opportunities")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        region_summary = summarize_closed(filtered, "regional_office")
        fig = px.bar(
            region_summary.sort_values("won_revenue", ascending=True),
            x="won_revenue",
            y="regional_office",
            orientation="h",
            text="won_revenue",
            title="Won Revenue by Region",
            labels={"regional_office": "Region", "won_revenue": "Won Revenue"},
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(height=420, xaxis_tickprefix="$", xaxis_title="Won Revenue", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        product_summary = summarize_closed(filtered, "product")
        fig = px.bar(
            product_summary.sort_values("won_revenue", ascending=True),
            x="won_revenue",
            y="product",
            orientation="h",
            text="win_rate",
            title="Won Revenue by Product with Win Rate Labels",
            labels={"product": "Product", "won_revenue": "Won Revenue", "win_rate": "Win Rate"},
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig.update_layout(height=420, xaxis_tickprefix="$", xaxis_title="Won Revenue", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        closed_stage = filtered[filtered["is_closed"]]
        win_loss = closed_stage.groupby("deal_stage")["opportunity_id"].nunique().reset_index(name="deals")
        fig = px.pie(
            win_loss,
            names="deal_stage",
            values="deals",
            hole=0.45,
            title="Closed Deal Outcomes",
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Research Questions Covered")
    rq1, rq2, rq3, rq4 = st.columns(4)
    rq1.info("How does performance differ by region, manager, and sales agent?")
    rq2.info("Which products create the strongest win rates and revenue contribution?")
    rq3.info("Are there quarter-over-quarter changes in revenue, volume, win rate, or cycle time?")
    rq4.info("Do account sectors, locations, or size segments align with stronger outcomes?")

# -----------------------------
# 2. Team Performance
# -----------------------------
with tab_team:
    st.subheader("Team Performance")
    st.write("Compare performance across regions, managers, or individual sales agents using closed-deal KPIs.")

    level = st.radio("Comparison level", ["Region", "Manager", "Sales Agent"], horizontal=True)
    if level == "Region":
        group_cols = ["regional_office"]
        label_col = "regional_office"
    elif level == "Manager":
        group_cols = ["manager", "regional_office"]
        label_col = "manager"
    else:
        group_cols = ["sales_agent", "manager", "regional_office"]
        label_col = "sales_agent"

    perf = summarize_closed(filtered, group_cols)

    if perf.empty:
        st.warning("No closed deals are available in the selected view.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                perf.sort_values("won_revenue", ascending=True).tail(20),
                x="won_revenue",
                y=label_col,
                orientation="h",
                text="won_revenue",
                title=f"Won Revenue by {level}",
                hover_data=["closed_opportunities", "won_deals", "win_rate", "avg_days_to_close"],
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(height=520, xaxis_tickprefix="$", yaxis_title="", xaxis_title="Won Revenue")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                perf,
                x="win_rate",
                y="won_revenue",
                size="closed_opportunities",
                color="regional_office" if "regional_office" in perf.columns else None,
                hover_name=label_col,
                hover_data=["closed_opportunities", "avg_won_deal", "avg_days_to_close"],
                title="Win Rate vs. Won Revenue",
            )
            fig.update_layout(height=520, xaxis_tickformat=".0%", yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)

        cols = group_cols + [
            "closed_opportunities",
            "won_deals",
            "lost_deals",
            "win_rate",
            "won_revenue",
            "revenue_share",
            "avg_won_deal",
            "revenue_per_closed_opp",
            "avg_days_to_close",
        ]
        st.dataframe(table_format(perf[cols]), use_container_width=True, hide_index=True)
        st.download_button(
            "Download team performance table",
            data=perf[cols].to_csv(index=False).encode("utf-8"),
            file_name="team_performance_summary.csv",
            mime="text/csv",
        )

# -----------------------------
# 3. Agent Diagnostics
# -----------------------------
with tab_agent:
    st.subheader("Agent Diagnostics")
    st.write("This page flags possible coaching opportunities using descriptive benchmarks. It is not a predictive model.")

    min_closed = st.slider(
        "Minimum closed opportunities for agent diagnostic",
        min_value=1,
        max_value=100,
        value=25,
        step=1,
    )

    agent_perf = summarize_closed(filtered, ["sales_agent", "manager", "regional_office"], min_closed=min_closed)
    agent_perf = add_agent_flag(agent_perf)

    if agent_perf.empty:
        st.warning("No agents meet the selected threshold.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            flags = agent_perf["diagnostic_flag"].value_counts().reset_index()
            flags.columns = ["diagnostic_flag", "agents"]
            fig = px.bar(
                flags,
                x="diagnostic_flag",
                y="agents",
                text="agents",
                title="Agent Diagnostic Flag Counts",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=450, xaxis_title="", yaxis_title="Agents")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                agent_perf,
                x="win_rate",
                y="revenue_per_closed_opp",
                size="closed_opportunities",
                color="diagnostic_flag",
                hover_name="sales_agent",
                hover_data=["manager", "regional_office", "won_revenue", "avg_days_to_close"],
                title="Agent Conversion vs. Revenue Efficiency",
            )
            fig.update_layout(height=450, xaxis_tickformat=".0%", yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)

        review_agents = agent_perf[agent_perf["diagnostic_flag"].str.startswith("Review")].sort_values(
            ["win_rate", "revenue_per_closed_opp"], ascending=[True, True]
        )

        st.markdown("#### Agents prioritized for review")
        if review_agents.empty:
            st.success("No agents are flagged for review under the current filters and threshold.")
        else:
            cols = [
                "sales_agent",
                "manager",
                "regional_office",
                "closed_opportunities",
                "win_rate",
                "won_revenue",
                "revenue_per_closed_opp",
                "avg_days_to_close",
                "diagnostic_flag",
            ]
            st.dataframe(table_format(review_agents[cols]), use_container_width=True, hide_index=True)

        st.markdown("#### Full agent diagnostic table")
        all_cols = [
            "sales_agent",
            "manager",
            "regional_office",
            "closed_opportunities",
            "won_deals",
            "lost_deals",
            "win_rate",
            "won_revenue",
            "avg_won_deal",
            "revenue_per_closed_opp",
            "avg_days_to_close",
            "diagnostic_flag",
        ]
        st.dataframe(table_format(agent_perf[all_cols]), use_container_width=True, hide_index=True)
        st.download_button(
            "Download agent diagnostic table",
            data=agent_perf[all_cols].to_csv(index=False).encode("utf-8"),
            file_name="agent_diagnostic_summary.csv",
            mime="text/csv",
        )

# -----------------------------
# 4. Product Analysis
# -----------------------------
with tab_product:
    st.subheader("Product Analysis")
    st.write("Compare products by revenue contribution, win rate, deal value, and price realization.")

    product_perf = summarize_closed(filtered, ["product", "series"])

    if product_perf.empty:
        st.warning("No closed deals are available for product analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                product_perf,
                x="win_rate",
                y="won_revenue",
                size="closed_opportunities",
                color="series",
                hover_name="product",
                hover_data=["closed_opportunities", "avg_won_deal", "avg_price_realization", "avg_days_to_close"],
                title="Product Win Rate vs. Won Revenue",
            )
            fig.update_layout(height=460, xaxis_tickformat=".0%", yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            prod_stage = stage_mix(filtered, "product")
            fig = px.bar(
                prod_stage,
                x="product",
                y="opportunities",
                color="deal_stage",
                title="Product Pipeline Stage Mix",
            )
            fig.update_layout(height=460, xaxis_title="", yaxis_title="Opportunities")
            st.plotly_chart(fig, use_container_width=True)

        cols = [
            "product",
            "series",
            "closed_opportunities",
            "won_deals",
            "lost_deals",
            "win_rate",
            "won_revenue",
            "revenue_share",
            "avg_won_deal",
            "avg_price_realization",
            "avg_days_to_close",
        ]
        st.dataframe(table_format(product_perf[cols]), use_container_width=True, hide_index=True)
        st.info(
            "Interpretation: a product with the highest win rate is not automatically the best commercial priority. Compare win rate, revenue contribution, volume, and average won deal size together."
        )

# -----------------------------
# 5. Quarterly Trends
# -----------------------------
with tab_trends:
    st.subheader("Quarterly Trends")
    st.write("Quarterly KPIs use close date and only include completed Won/Lost opportunities.")

    closed = filtered[filtered["is_closed"]].dropna(subset=["close_quarter"]).copy()

    if closed.empty:
        st.warning("No closed deals are available for trend analysis.")
    else:
        q = (
            closed.groupby("close_quarter")
            .agg(
                closed_opportunities=("opportunity_id", "nunique"),
                won_deals=("won_flag", "sum"),
                lost_deals=("lost_flag", "sum"),
                won_revenue=("won_value", "sum"),
                avg_close_value=("close_value", "mean"),
                avg_days_to_close=("days_to_close", "mean"),
            )
            .reset_index()
            .sort_values("close_quarter")
        )
        q["won_deals"] = q["won_deals"].astype(int)
        q["lost_deals"] = q["lost_deals"].astype(int)
        q["win_rate"] = safe_divide(q["won_deals"], q["closed_opportunities"])

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                q,
                x="close_quarter",
                y="won_revenue",
                text="won_revenue",
                title="Won Revenue by Close Quarter",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(height=430, xaxis_title="Close Quarter", yaxis_title="Won Revenue", yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.line(
                q,
                x="close_quarter",
                y="win_rate",
                markers=True,
                title="Win Rate by Close Quarter",
            )
            fig.update_layout(height=430, xaxis_title="Close Quarter", yaxis_title="Win Rate", yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.bar(
                q,
                x="close_quarter",
                y="closed_opportunities",
                text="closed_opportunities",
                title="Closed Opportunity Volume by Quarter",
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=400, xaxis_title="Close Quarter", yaxis_title="Closed Opportunities")
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = px.line(
                q,
                x="close_quarter",
                y="avg_days_to_close",
                markers=True,
                title="Average Days to Close by Quarter",
            )
            fig.update_layout(height=400, xaxis_title="Close Quarter", yaxis_title="Avg Days to Close")
            st.plotly_chart(fig, use_container_width=True)

        cols = [
            "close_quarter",
            "closed_opportunities",
            "won_deals",
            "lost_deals",
            "win_rate",
            "won_revenue",
            "avg_close_value",
            "avg_days_to_close",
        ]
        st.dataframe(table_format(q[cols]), use_container_width=True, hide_index=True)

# -----------------------------
# 6. Account Segments
# -----------------------------
with tab_segments:
    st.subheader("Account Segment Analysis")
    st.write("Analyze whether sector, company size, revenue size, or geography aligns with stronger sales outcomes.")

    segment_map = {
        "Sector": "sector_display",
        "Employee Segment": "employee_segment",
        "Revenue Segment": "revenue_segment",
        "Office Location": "office_location_display",
    }
    segment_label = st.selectbox("Segmentation lens", list(segment_map.keys()))
    segment_col = segment_map[segment_label]

    seg = summarize_closed(filtered, segment_col)

    if seg.empty:
        st.warning("No closed deals are available for account segment analysis.")
    else:
        top_seg = seg.sort_values("won_revenue", ascending=False).head(15)
        c1, c2 = st.columns(2)

        with c1:
            fig = px.bar(
                top_seg.sort_values("won_revenue", ascending=True),
                x="won_revenue",
                y=segment_col,
                orientation="h",
                text="won_revenue",
                title=f"Won Revenue by {segment_label}",
            )
            fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig.update_layout(height=500, xaxis_tickprefix="$", xaxis_title="Won Revenue", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                seg,
                x="win_rate",
                y="revenue_per_closed_opp",
                size="closed_opportunities",
                hover_name=segment_col,
                hover_data=["won_revenue", "avg_won_deal", "avg_days_to_close"],
                title=f"{segment_label}: Win Rate vs. Revenue Efficiency",
            )
            fig.update_layout(height=500, xaxis_tickformat=".0%", yaxis_tickprefix="$")
            st.plotly_chart(fig, use_container_width=True)

        cols = [
            segment_col,
            "closed_opportunities",
            "won_deals",
            "lost_deals",
            "win_rate",
            "won_revenue",
            "revenue_share",
            "avg_won_deal",
            "revenue_per_closed_opp",
            "avg_days_to_close",
        ]
        st.dataframe(table_format(seg[cols]), use_container_width=True, hide_index=True)

# -----------------------------
# 7. Data Explorer
# -----------------------------
with tab_data:
    st.subheader("Data Explorer")
    st.write("Inspect and export the filtered analytical dataset used by the dashboard.")

    display_cols = [
        "opportunity_id",
        "sales_agent",
        "manager",
        "regional_office",
        "product",
        "series",
        "account",
        "sector_display",
        "office_location_display",
        "employee_segment",
        "revenue_segment",
        "deal_stage",
        "engage_date",
        "close_date",
        "close_value",
        "days_to_close",
        "sales_price",
        "price_realization",
    ]

    sort_col = st.selectbox(
        "Sort by",
        ["close_value", "engage_date", "close_date", "days_to_close", "product", "sales_agent"],
    )
    ascending = st.checkbox("Sort ascending", value=False)

    explorer = filtered[display_cols].sort_values(sort_col, ascending=ascending, na_position="last")
    st.dataframe(table_format(explorer), use_container_width=True, hide_index=True, height=520)

    st.download_button(
        "Download filtered dataset",
        data=explorer.to_csv(index=False).encode("utf-8"),
        file_name="filtered_crm_sales_opportunities.csv",
        mime="text/csv",
    )

    st.markdown("#### Data quality rules used")
    c1, c2, c3 = st.columns(3)
    c1.metric("Open Opportunities", whole(int(filtered["is_open"].sum())))
    c1.caption("Included in pipeline volume, excluded from closed-deal KPIs.")
    c2.metric("Missing Account Link", whole(int(filtered["account"].isna().sum())))
    c2.caption("Kept for pipeline counts, not used for account interpretation.")
    c3.metric("GTX Pro Rows", whole(int(filtered["product"].eq("GTX Pro").sum())))
    c3.caption("Includes original `GTXPro` values standardized to `GTX Pro`.")


st.divider()
st.caption("CRM Sales Opportunities Dashboard | Descriptive analytics final project | Maven Analytics CRM Sales Opportunities dataset")
