"""
dashboard.py
============
Main Streamlit dashboard – all 11 Plotly charts, KPI cards, sidebar filters,
business insights section, and ML forecasting panel.

Run with:
    streamlit run dashboard.py

Author : Sales Analytics Dashboard
Purpose: Portfolio / Internship project
"""

import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout so Unicode chars in print() don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Internal modules ─────────────────────────────────────────────────────────
from modules.data_loader         import load_raw_data
from modules.data_cleaning       import clean_data
from modules.feature_engineering import (
    engineer_features,
    build_monthly_sales,
    build_region_summary,
    build_product_summary,
    build_pivot_region_product,
    build_customer_summary,
    build_payment_summary,
    build_promotion_summary,
    build_return_heatmap,
    build_discount_revenue,
)
from modules.forecasting import train_forecast_model, forecast_future, analyze_product_growth
from modules.utils import (
    compute_kpis,
    generate_insights,
    generate_root_cause_analysis,
    fmt_currency,
    fmt_pct,
    fmt_number,
    get_chart_layout,
    PALETTE,
    REGION_COLORS,
    PRODUCT_COLORS,
    get_theme_css,
)
from modules.intelligence import (
    generate_executive_summary,
    get_top_movers,
    detect_anomalies,
    run_what_if,
    build_excel_export,
)
from modules.explorer import render_business_explorer

# Strictly enforce dark mode
st.session_state.theme = "dark"

CHART_LAYOUT = get_chart_layout(st.session_state.theme)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading & processing data…")
def get_processed_data():
    raw   = load_raw_data()
    clean = clean_data(raw)
    feat  = engineer_features(clean)
    return feat


@st.cache_data(show_spinner="🤖 Training forecast model…")
def get_forecast(_df):
    result = train_forecast_model(_df)
    future = forecast_future(result, n_months=6)
    return result, future


df_full = get_processed_data()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Filters (Enterprise Batch Workflow)
# ─────────────────────────────────────────────────────────────────────────────

# Initialize Batch Filter States
if "applied_filters" not in st.session_state:
    st.session_state.applied_filters = {
        "region":         sorted(df_full["region"].dropna().unique()),
        "product":        sorted(df_full["product"].dropna().unique()),
        "customer_type":  sorted(df_full["customer_type"].dropna().unique()),
        "payment_method": sorted(df_full["payment_method"].dropna().unique()),
        "promotion_used": sorted(df_full["promotion_used"].dropna().unique()),
    }

if "pending_filters" not in st.session_state:
    st.session_state.pending_filters = {
        k: list(v) for k, v in st.session_state.applied_filters.items()
    }

def check_pending_changes():
    for k in st.session_state.applied_filters:
        if set(st.session_state.pending_filters.get(k, [])) != set(st.session_state.applied_filters[k]):
            return True
    return False

def get_changed_filters():
    labels = {
        "region": "Region", "product": "Product",
        "customer_type": "Customer Type",
        "payment_method": "Payment", "promotion_used": "Promotion",
    }
    changed = {}
    for k in st.session_state.applied_filters:
        p = set(st.session_state.pending_filters.get(k, []))
        a = set(st.session_state.applied_filters[k])
        if p != a:
            changed[k] = {"label": labels[k], "added": p - a, "removed": a - p}
    return changed

with st.sidebar:
    has_pending     = check_pending_changes()
    changed_filters = get_changed_filters() if has_pending else {}

    # ── FLOATING APPLY BANNER ─────────────────────────────────────────────────
    if has_pending:
        diff_parts = []
        for k, info in changed_filters.items():
            if info["removed"] and info["added"]:
                diff_parts.append(f"<b>{info['label']}</b>: modified")
            elif info["removed"]:
                diff_parts.append(f"<b>{info['label']}</b>: &minus;{len(info['removed'])}")
            elif info["added"]:
                diff_parts.append(f"<b>{info['label']}</b>: +{len(info['added'])}")
        diff_html = " &nbsp;·&nbsp; ".join(diff_parts)

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 12px;
            box-shadow: 0 0 0 2px rgba(108,99,255,0.5), 0 6px 24px rgba(108,99,255,0.4);
            animation: filterPulse 2s ease-in-out infinite;
        ">
            <div style="font-size:0.78rem; color:rgba(255,255,255,0.9); font-weight:600; margin-bottom:4px;">
                ✏️ Unsaved changes
            </div>
            <div style="font-size:0.72rem; color:rgba(255,255,255,0.7); line-height:1.6;">
                {diff_html}
            </div>
        </div>
        <style>
        @keyframes filterPulse {{
            0%,100% {{ box-shadow: 0 0 0 2px rgba(108,99,255,0.5), 0 6px 24px rgba(108,99,255,0.3); }}
            50%      {{ box-shadow: 0 0 0 3px rgba(108,99,255,0.8), 0 6px 32px rgba(108,99,255,0.55); }}
        }}
        </style>
        """, unsafe_allow_html=True)

        ca, cr = st.columns(2)
        if ca.button("✅ Apply", use_container_width=True, type="primary", key="apply_top"):
            st.session_state.applied_filters = {k: list(v) for k, v in st.session_state.pending_filters.items()}
            st.rerun()
        if cr.button("↩️ Revert", use_container_width=True, key="revert_top"):
            st.session_state.pending_filters = {k: list(v) for k, v in st.session_state.applied_filters.items()}
            st.rerun()

        st.markdown("<hr style='margin:10px 0; border-color:rgba(108,99,255,0.2);'>", unsafe_allow_html=True)

    # ── Header with status badge ──────────────────────────────────────────────
    if has_pending:
        badge = "<span style='font-size:0.68rem; background:rgba(124,58,237,0.2); color:#A78BFA; border:1px solid rgba(124,58,237,0.4); border-radius:6px; padding:1px 7px; vertical-align:middle; font-weight:600;'>UNSAVED</span>"
    else:
        badge = "<span style='font-size:0.68rem; background:rgba(46,204,113,0.15); color:#2ECC71; border:1px solid rgba(46,204,113,0.35); border-radius:6px; padding:1px 7px; vertical-align:middle; font-weight:600;'>✓ APPLIED</span>"
    st.markdown(f"### 🎛️ Filters &nbsp;{badge}", unsafe_allow_html=True)

    # ── Date range ────────────────────────────────────────────────────────────
    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("<div style='font-size:0.72rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:0.07em; font-weight:600; margin:10px 0 4px 0;'>Category Filters</div>", unsafe_allow_html=True)

    # ── Pill filter builder ───────────────────────────────────────────────────
    def render_pill_filter(category_col, label, icon):
        options      = sorted(df_full[category_col].dropna().unique())
        pending_vals = st.session_state.pending_filters.get(category_col, list(options))
        is_changed   = category_col in changed_filters

        n_sel   = len(pending_vals)
        n_total = len(options)
        count_badge = f"{n_sel}/{n_total}"

        # Diff badge on the expander label
        diff_badge = ""
        if is_changed:
            info = changed_filters[category_col]
            if info["removed"]:
                diff_badge = f" 🔴−{len(info['removed'])}"
            if info["added"]:
                diff_badge += f" 🟢+{len(info['added'])}"

        with st.expander(f"{icon} {label} ({count_badge}){diff_badge}", expanded=is_changed):
            # Inline diff hint
            if is_changed:
                info = changed_filters[category_col]
                parts = []
                for v in info["removed"]:
                    parts.append(f"<span style='color:#FF6584;'>−&nbsp;{v}</span>")
                for v in info["added"]:
                    parts.append(f"<span style='color:#43E8D8;'>+&nbsp;{v}</span>")
                st.markdown(
                    "<div style='font-size:0.72rem; padding:5px 8px; background:rgba(108,99,255,0.1);"
                    "border-left:3px solid #6C63FF; border-radius:4px; margin-bottom:8px; line-height:1.9;'>"
                    + " &nbsp; ".join(parts) + "</div>",
                    unsafe_allow_html=True
                )

            c1, c2 = st.columns(2)
            if c1.button("All", key=f"all_{category_col}", use_container_width=True):
                st.session_state.pending_filters[category_col] = list(options)
                st.rerun()
            if c2.button("None", key=f"clear_{category_col}", use_container_width=True):
                st.session_state.pending_filters[category_col] = []
                st.rerun()

            sel = st.pills(
                "Select",
                options=options,
                default=pending_vals,
                selection_mode="multi",
                label_visibility="collapsed",
                key=f"pill_{category_col}"
            )
            if list(sel) != list(pending_vals):
                st.session_state.pending_filters[category_col] = list(sel)
                st.rerun()

    render_pill_filter("region",         "Region",         "🗺️")
    render_pill_filter("product",        "Product",        "📦")
    render_pill_filter("customer_type",  "Customer Type",  "👥")
    render_pill_filter("payment_method", "Payment Method", "💳")
    render_pill_filter("promotion_used", "Promotion",      "🎁")

    st.markdown("<hr style='margin:12px 0; border-color:rgba(108,99,255,0.2);'>", unsafe_allow_html=True)

    if st.button("🔄 Reset All to Defaults", use_container_width=True, key="reset_all"):
        default = {k: sorted(df_full[k].dropna().unique()) for k in st.session_state.applied_filters}
        st.session_state.pending_filters = {k: list(v) for k, v in default.items()}
        st.session_state.applied_filters = {k: list(v) for k, v in default.items()}
        st.rerun()

    st.markdown(
        "<div style='color:#7F8C8D; font-size:0.72rem; margin-top:8px;'>"
        "Sales Analytics Dashboard v2.0<br>Premium BI Edition"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Cross-Filtering Logic (Advanced Analytics UX)
# ─────────────────────────────────────────────────────────────────────────────
cross_region = None
if "region_chart" in st.session_state:
    sel = st.session_state["region_chart"].get("selection", {}).get("points", [])
    if sel:
        # Pie chart selections usually contain 'label'
        cross_region = [p.get("label", p.get("x")) for p in sel if p.get("label", p.get("x"))]

cross_product = None
if "product_chart" in st.session_state:
    sel = st.session_state["product_chart"].get("selection", {}).get("points", [])
    if sel:
        # Bar chart selections contain 'x'
        cross_product = [p.get("x") for p in sel if p.get("x")]

if cross_region or cross_product:
    st.markdown('<div class="sticky-header" style="background: rgba(108,99,255,0.1); padding: 8px 16px; font-size: 0.9rem; border-left: 4px solid #6C63FF; margin-top: -15px;">'
                '<b>🔍 Active Drill-down:</b> ' + 
                (f'Region = {", ".join(cross_region)} ' if cross_region else '') + 
                (f'Product = {", ".join(cross_product)}' if cross_product else '') + 
                ' <span style="color:var(--text-secondary); font-size:0.8rem; margin-left: 10px;">(Click empty space on chart to clear)</span></div>', 
                unsafe_allow_html=True)

# Apply Date Filters
if len(date_range) == 2:
    start_date, end_date = date_range
    df = df_full[
        (df_full["date"].dt.date >= start_date) &
        (df_full["date"].dt.date <= end_date)
    ].copy()
else:
    df = df_full.copy()

# Apply Batch Filters
for col, vals in st.session_state.applied_filters.items():
    if vals:
        df = df[df[col].isin(vals)]

# Apply Cross-Filters
if cross_region:  df = df[df["region"].isin(cross_region)]
if cross_product: df = df[df["product"].isin(cross_product)]

if df.empty:
    st.error("⚠️ No data matches your current filter selection. "
             "Please broaden the filters.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Header & Theme Toggle (Command Center)
# ─────────────────────────────────────────────────────────────────────────────
kpis = compute_kpis(df)

import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

is_dark = st.session_state.theme == "dark"
theme_icon = "🌙" if is_dark else "☀️"

header_html = f"""
<div class="command-header">
    <div>
        <h1 class="command-title">Sales Intelligence Platform</h1>
        <p class="command-subtitle">Executive Command Center · {len(df):,} records active</p>
    </div>
    <div style="display: flex; gap: 32px; align-items: center;">
        <div style="text-align: right;">
            <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Health Score</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: var(--positive);">{kpis.get('health_score', 0):.0f}/100</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">Growth Score</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: var(--primary);">{kpis.get('growth_score', 0):.0f}/100</div>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# (Theme Toggle disabled to strictly enforce Premium Dark Mode aesthetic)

# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span>🔢 Platform Metrics</span></div>', unsafe_allow_html=True)

def render_delta(delta: float):
    if delta > 0:
        return f'<div class="delta-badge positive">▲ +{delta:.1f}%</div>'
    elif delta < 0:
        return f'<div class="delta-badge negative">▼ {delta:.1f}%</div>'
    return f'<div class="delta-badge neutral">➖ 0.0%</div>'

# Level 1 KPIs
l1_cols = st.columns(3, gap="medium")
l1_data = [
    ("Revenue", fmt_currency(kpis.get("total_revenue", 0)), kpis.get("revenue_delta", 0), "💰", "vs prior period"),
    ("Profit", fmt_currency(kpis.get("total_profit", 0)), kpis.get("profit_delta", 0), "💎", "vs prior period"),
    ("Orders", fmt_number(kpis.get("total_orders", 0)), kpis.get("orders_delta", 0), "🛒", "vs prior period"),
]

for col, (title, val, delta, icon, sub) in zip(l1_cols, l1_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <div class="kpi-title">{title}</div>
                <div class="kpi-icon-wrap">{icon}</div>
            </div>
            <div class="kpi-hero-value level-1">{val}</div>
            <div class="kpi-footer">
                {render_delta(delta)}
                <span class="kpi-subtext">{sub}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Level 2 KPIs
l2_cols = st.columns(4, gap="medium")
l2_data = [
    ("Qty Sold", fmt_number(kpis.get("total_quantity", 0)), kpis.get("qty_delta", 0), "📦"),
    ("AOV", fmt_currency(kpis.get("avg_order_value", 0)), kpis.get("aov_delta", 0), "💵"),
    ("Return Rate", fmt_pct(kpis.get("return_rate", 0)), kpis.get("rr_delta", 0), "↩️"),
    ("Best Region", kpis.get("best_region", "N/A"), None, "🏆"),
]

for col, (title, val, delta, icon) in zip(l2_cols, l2_data):
    with col:
        footer_html = f'<div class="kpi-footer">{render_delta(delta)}<span class="kpi-subtext">vs prior period</span></div>' if delta is not None else '<div class="kpi-footer"><span class="kpi-subtext">Top performer</span></div>'
        st.markdown(f"""
        <div class="kpi-card" style="padding: 18px;">
            <div class="kpi-header" style="margin-bottom: 12px;">
                <div class="kpi-title" style="font-size: 0.75rem;">{title}</div>
                <div class="kpi-icon-wrap" style="padding: 4px 6px; font-size: 0.9rem;">{icon}</div>
            </div>
            <div class="kpi-hero-value" style="font-size: 1.6rem; margin-bottom: 12px;">{val}</div>
            {footer_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
(tab_overview, tab_exec, tab_explorer, tab_movers,
 tab_whatif, tab_anomaly, tab_export) = st.tabs([
    "📊 Overview",
    "🧠 Executive Summary",
    "🔭 Business Explorer",
    "📈 Top Movers",
    "🎛️ What-If",
    "⚠️ Anomalies",
    "📤 Export",
])

with tab_overview:

    # ─────────────────────────────────────────────────────────────────────────────
    ai_story = generate_root_cause_analysis(df)
    st.markdown(f"""
    <div style="
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        backdrop-filter: blur(12px);
    ">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:1.5rem;">🤖</span>
            <h3 style="margin:0; font-size:1.1rem; color:var(--text-primary);">AI Analyst Root Cause Analysis</h3>
        </div>
        <div style="font-size:0.95rem; color:var(--text-secondary); line-height:1.6;">
            {ai_story}
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # Helper: apply chart layout
    # ─────────────────────────────────────────────────────────────────────────────
    def apply_layout(fig):
        fig.update_layout(**CHART_LAYOUT)
        return fig


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 1 – Monthly Sales Trend
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Sales Trends & Regional Performance</div>',
                unsafe_allow_html=True)

    monthly = build_monthly_sales(df)
    c1, c2  = st.columns([3, 2], gap="medium")

    with c1:
        ct, cs = st.columns([3, 2], vertical_alignment="center")
        with ct:
            st.markdown("<h4 style='margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--text-primary);'>📅 Monthly Sales Trend</h4>", unsafe_allow_html=True)
        with cs:
            viz_monthly = st.selectbox(
                "Type", 
                ["Line + Bar (Recommended)", "Area Chart", "Line Chart", "Bar Chart"], 
                label_visibility="collapsed", 
                key="viz_monthly"
            )
        
        fig_monthly = go.Figure()
    
        if viz_monthly == "Area Chart":
            fig_monthly.add_trace(go.Scatter(
                x=monthly["year_month"], y=monthly["total_revenue"],
                mode="lines", fill="tozeroy", name="Net Revenue",
                line=dict(color="#6C63FF", width=2.5),
                fillcolor="rgba(108,99,255,0.2)",
                hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>"
            ))
            fig_monthly.update_layout(**CHART_LAYOUT, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
    
        elif viz_monthly == "Line Chart":
            fig_monthly.add_trace(go.Scatter(
                x=monthly["year_month"], y=monthly["total_revenue"],
                mode="lines+markers", name="Net Revenue",
                line=dict(color="#6C63FF", width=2.5),
                marker=dict(size=8, color="#6C63FF", line=dict(width=1.5, color="#FFFFFF")),
                hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>"
            ))
            fig_monthly.update_layout(**CHART_LAYOUT, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        elif viz_monthly == "Bar Chart":
            fig_monthly = px.bar(
                monthly, x="year_month", y="total_revenue",
                color_discrete_sequence=["#6C63FF"]
            )
            fig_monthly.update_traces(hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>")
            fig_monthly.update_layout(**CHART_LAYOUT, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        else:
            # Default Line+Bar Combo
            fig_monthly.add_trace(go.Scatter(
                x=monthly["year_month"], y=monthly["total_revenue"],
                mode="lines+markers", name="Net Revenue",
                line=dict(color="#6C63FF", width=2.5),
                marker=dict(size=6, color="#6C63FF", line=dict(width=1.5, color="#FFFFFF")),
                fill="tozeroy", fillcolor="rgba(108,99,255,0.12)",
                hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
            ))
            fig_monthly.add_trace(go.Bar(
                x=monthly["year_month"], y=monthly["total_orders"],
                name="Orders", marker_color="rgba(67,232,216,0.35)",
                yaxis="y2", hovertemplate="<b>%{x}</b><br>Orders: %{y}<extra></extra>",
            ))
            fig_monthly.update_layout(
                **CHART_LAYOUT,
                yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis2=dict(title="Order Count", overlaying="y", side="right", gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(tickangle=-45, gridcolor="rgba(255,255,255,0.04)"),
            )
        
        st.plotly_chart(fig_monthly, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 2 – Revenue by Region (Donut + Bar)
    # ─────────────────────────────────────────────────────────────────────────────
    with c2:
        ct, cs = st.columns([2, 2], vertical_alignment="center")
        with ct:
            st.markdown("<h4 style='margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--text-primary);'>🗺️ Revenue by Region</h4>", unsafe_allow_html=True)
        with cs:
            viz_region = st.selectbox(
                "Type", 
                ["Donut (Recommended)", "Pie Chart", "Treemap", "Bar Chart"], 
                label_visibility="collapsed", 
                key="viz_region"
            )
        
        region_df = build_region_summary(df)
    
        if viz_region == "Pie Chart":
            fig_region = go.Figure(go.Pie(
                labels=region_df["region"], values=region_df["total_revenue"],
                hole=0, marker=dict(colors=[REGION_COLORS.get(r, "#6C63FF") for r in region_df["region"]], line=dict(color="#1A1D27", width=1)),
                textinfo="label+percent", hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>",
            ))
            fig_region.update_layout(**CHART_LAYOUT)
        
        elif viz_region == "Treemap":
            fig_region = px.treemap(
                region_df, path=[px.Constant("Regions"), "region"], values="total_revenue",
                color="region", color_discrete_map=REGION_COLORS
            )
            fig_region.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>")
            fig_region.update_layout(**CHART_LAYOUT)
        
        elif viz_region == "Bar Chart":
            fig_region = px.bar(
                region_df.sort_values("total_revenue", ascending=False), 
                x="region", y="total_revenue",
                color="region", color_discrete_map=REGION_COLORS
            )
            fig_region.update_traces(hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>")
            fig_region.update_layout(**CHART_LAYOUT, showlegend=False, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))

        else:
            # Default Donut
            fig_region = go.Figure(go.Pie(
                labels=region_df["region"],
                values=region_df["total_revenue"],
                hole=0.55,
                marker=dict(
                    colors=[REGION_COLORS.get(r, "#6C63FF") for r in region_df["region"]],
                    line=dict(color="#1A1D27", width=2),
                ),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}"
                              "<br>Share: %{percent}<extra></extra>",
                textfont=dict(size=11),
            ))
            fig_region.update_layout(
                **CHART_LAYOUT,
                annotations=[dict(
                    text=f"<b>{fmt_currency(region_df['total_revenue'].sum())}</b>",
                    x=0.5, y=0.5, font_size=16, showarrow=False,
                    font=dict(color="var(--text-primary)"),
                )],
                legend=dict(orientation="v", x=1.02, y=0.5),
                showlegend=True,
                clickmode="event+select",
                hovermode="closest",
            )
        
        # Cross filtering selection handling
        st.plotly_chart(fig_region, width="stretch", key="region_chart", on_select="rerun", selection_mode="points")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 3 – Product Performance Analysis (Treemap)
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📦 Product & Customer Analysis</div>',
                unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        ct, cs = st.columns([2, 2], vertical_alignment="center")
        with ct:
            st.markdown("<h4 style='margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--text-primary);'>📦 Product Performance</h4>", unsafe_allow_html=True)
        with cs:
            viz_product = st.selectbox(
                "Type", 
                ["Treemap (Recommended)", "Bar Chart", "Horizontal Bar Chart"], 
                label_visibility="collapsed", 
                key="viz_product"
            )
        
        prod_df  = build_product_summary(df)
        prod_df_filtered = prod_df[prod_df["total_revenue"] > 0]
    
        if viz_product == "Bar Chart":
            fig_prod = px.bar(
                prod_df_filtered.sort_values("total_revenue", ascending=False),
                x="product", y="total_revenue", color="return_rate",
                color_continuous_scale=[[0, "#43E8D8"], [0.5, "#6C63FF"], [1, "#FF6584"]]
            )
            fig_prod.update_traces(hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<br>Return Rate: %{color:.1%}<extra></extra>")
            fig_prod.update_layout(**CHART_LAYOUT, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        elif viz_product == "Horizontal Bar Chart":
            fig_prod = px.bar(
                prod_df_filtered.sort_values("total_revenue", ascending=True),
                y="product", x="total_revenue", color="return_rate", orientation="h",
                color_continuous_scale=[[0, "#43E8D8"], [0.5, "#6C63FF"], [1, "#FF6584"]]
            )
            fig_prod.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<br>Return Rate: %{color:.1%}<extra></extra>")
            fig_prod.update_layout(**CHART_LAYOUT, xaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        else:
            # Default Treemap
            fig_prod = px.treemap(
                prod_df_filtered,
                path=[px.Constant("Products"), "product"],
                values="total_revenue",
                color="return_rate",
                color_continuous_scale=[[0, "#43E8D8"], [0.5, "#6C63FF"], [1, "#FF6584"]]
            )
            fig_prod.update_traces(
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Return Rate: %{color:.1%}<extra></extra>",
                textinfo="label+value+percent parent",
                marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)"))
            )
            fig_prod.update_layout(**CHART_LAYOUT)
        
        # Using selection_mode for Cross Filtering
        st.plotly_chart(fig_prod, width="stretch", key="product_chart", on_select="rerun", selection_mode="points")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 4 – Customer Analysis (Sunburst)
    # ─────────────────────────────────────────────────────────────────────────────
    with c4:
        ct, cs = st.columns([2, 2], vertical_alignment="center")
        with ct:
            st.markdown("<h4 style='margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--text-primary);'>👥 Customer Demographics</h4>", unsafe_allow_html=True)
        with cs:
            viz_cust = st.selectbox(
                "Type", 
                ["Sunburst (Recommended)", "Treemap", "Bar Chart"], 
                label_visibility="collapsed", 
                key="viz_cust"
            )
        
        cust_region_df = df.groupby(["region", "customer_type"]).agg({"net_revenue": "sum"}).reset_index()
        cust_region_df = cust_region_df[cust_region_df["net_revenue"] > 0]
    
        if viz_cust == "Treemap":
            fig_cust = px.treemap(
                cust_region_df,
                path=["region", "customer_type"],
                values="net_revenue",
                color="net_revenue",
                color_continuous_scale=[[0, "#6C63FF"], [0.5, "#43E8D8"], [1, "#FF6584"]]
            )
            fig_cust.update_traces(
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>",
                marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)"))
            )
            fig_cust.update_layout(**CHART_LAYOUT)
        
        elif viz_cust == "Bar Chart":
            flat_cust_df = df.groupby("customer_type").agg({"net_revenue": "sum"}).reset_index()
            fig_cust = px.bar(
                flat_cust_df.sort_values("net_revenue", ascending=False),
                x="customer_type", y="net_revenue", color="customer_type",
                color_discrete_sequence=["#6C63FF", "#43E8D8", "#FF6584", "#F39C12"]
            )
            fig_cust.update_traces(hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>")
            fig_cust.update_layout(**CHART_LAYOUT, showlegend=False, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        else:
            # Default Sunburst
            fig_cust = px.sunburst(
                cust_region_df,
                path=["region", "customer_type"],
                values="net_revenue",
                color="net_revenue",
                color_continuous_scale=[[0, "#6C63FF"], [0.5, "#43E8D8"], [1, "#FF6584"]]
            )
            fig_cust.update_traces(
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percentParent:.1%}<extra></extra>",
                marker=dict(line=dict(width=1, color="rgba(255,255,255,0.2)"))
            )
            fig_cust.update_layout(**CHART_LAYOUT)
        
        st.plotly_chart(fig_cust, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 5 & 6 – Payment Method + Promotion Effectiveness
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💳 Payment & Promotions</div>',
                unsafe_allow_html=True)

    c5, c6 = st.columns(2, gap="medium")

    with c5:
        ct, cs = st.columns([2, 2], vertical_alignment="center")
        with ct:
            st.markdown("<h4 style='margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--text-primary);'>💳 Payment Method Analysis</h4>", unsafe_allow_html=True)
        with cs:
            viz_pay = st.selectbox(
                "Type", 
                ["Bar Chart (Recommended)", "Horizontal Bar Chart", "Donut Chart", "Pie Chart"], 
                label_visibility="collapsed", 
                key="viz_pay"
            )
        
        pay_df  = build_payment_summary(df)
    
        if viz_pay == "Donut Chart":
            fig_pay = px.pie(pay_df, values="total_revenue", names="payment_method", hole=0.5, color_discrete_sequence=PRODUCT_COLORS)
            fig_pay.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>")
            fig_pay.update_layout(**CHART_LAYOUT)
        
        elif viz_pay == "Pie Chart":
            fig_pay = px.pie(pay_df, values="total_revenue", names="payment_method", color_discrete_sequence=PRODUCT_COLORS)
            fig_pay.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>")
            fig_pay.update_layout(**CHART_LAYOUT)
        
        elif viz_pay == "Horizontal Bar Chart":
            fig_pay = px.bar(pay_df.sort_values("total_revenue", ascending=True), y="payment_method", x="total_revenue", color="payment_method", color_discrete_sequence=PRODUCT_COLORS, orientation="h")
            fig_pay.update_traces(hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>")
            fig_pay.update_layout(**CHART_LAYOUT, showlegend=False, xaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        else:
            # Default Bar Chart
            fig_pay = px.bar(
                pay_df, x="payment_method", y="total_revenue", color="payment_method",
                text_auto=".2s", color_discrete_sequence=PRODUCT_COLORS
            )
            fig_pay.update_traces(
                textfont_size=11, marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
            )
            fig_pay.update_layout(**CHART_LAYOUT, showlegend=False, yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        
        st.plotly_chart(fig_pay, width="stretch")

    with c6:
        ct, cs = st.columns([2, 2], vertical_alignment="center")
        with ct:
            st.markdown("<h4 style='margin:0; padding:0; font-size:1.1rem; font-weight:600; color:var(--text-primary);'>🎁 Promotion Effectiveness</h4>", unsafe_allow_html=True)
        with cs:
            viz_promo = st.selectbox(
                "Type", 
                ["Waterfall (Recommended)", "Bar Chart"], 
                label_visibility="collapsed", 
                key="viz_promo"
            )
        
        promo_df  = build_promotion_summary(df)
    
        if viz_promo == "Bar Chart":
            fig_promo = px.bar(
                promo_df,
                x="description", y="total_revenue", color="avg_discount", text_auto=".2s",
                color_continuous_scale=[[0, "#6C63FF"], [0.5, "#43E8D8"], [1, "#FF6584"]],
                hover_data=["return_rate", "avg_quantity"],
            )
            fig_promo.update_traces(
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Revenue: ₹%{y:,.0f}<br>"
                    "Avg Discount: %{marker.color:.0%}<extra></extra>"
                ),
            )
            fig_promo.update_layout(
                **CHART_LAYOUT,
                coloraxis_colorbar=dict(title="Discount", tickformat=".0%"),
                yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)")
            )
        
        else:
            # Default Waterfall
            base_revenue = promo_df[promo_df["description"].str.contains("NONE|None|none", na=False)]["total_revenue"].sum()
            promo_filtered = promo_df[~promo_df["description"].str.contains("NONE|None|none", na=False)]
        
            x_data = ["Baseline (No Promo)"] + promo_filtered["description"].tolist() + ["Total"]
            y_data = [base_revenue] + promo_filtered["total_revenue"].tolist() + [0]
            measure = ["absolute"] + ["relative"] * len(promo_filtered) + ["total"]
        
            fig_promo = go.Figure(go.Waterfall(
                name="20", orientation="v", measure=measure, x=x_data, y=y_data, textposition="outside",
                text=[fmt_currency(y) for y in y_data], connector={"line":{"color":"#4B5563"}},
                decreasing={"marker":{"color":"#FF6584"}}, increasing={"marker":{"color":"#6C63FF"}}, totals={"marker":{"color":"#43E8D8"}}
            ))
            fig_promo.update_layout(
                **CHART_LAYOUT, 
                yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"), 
                showlegend=False
            )

        st.plotly_chart(fig_promo, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 7 – Discount vs Revenue Scatter
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Advanced Analysis</div>',
                unsafe_allow_html=True)

    c7, c8 = st.columns(2, gap="medium")

    with c7:
        scatter_df = build_discount_revenue(df)
        fig_scatter = px.scatter(
            scatter_df,
            x="discount_pct",
            y="net_revenue",
            color="product",
            size="quantity_sold",
            symbol="customer_type",
            opacity=0.75,
            color_discrete_sequence=PRODUCT_COLORS,
            title="💸 Discount % vs Net Revenue",
            labels={"discount_pct": "Discount (%)", "net_revenue": "Net Revenue (₹)"},
            hover_data=["region", "customer_type", "quantity_sold"],
        )
        fig_scatter.update_traces(
            marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
        )
        fig_scatter.update_layout(
            **CHART_LAYOUT,
            xaxis=dict(title="Discount (%)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="v", x=1.02),
        )
        st.plotly_chart(fig_scatter, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 8 – Return Rate Heatmap
    # ─────────────────────────────────────────────────────────────────────────────
    with c8:
        heatmap_df = build_return_heatmap(df)
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns.tolist(),
            y=heatmap_df.index.tolist(),
            colorscale=[
                [0.0,  "#1A1D27"],
                [0.3,  "#6C63FF"],
                [0.65, "#FF6584"],
                [1.0,  "#E74C3C"],
            ],
            text=[[f"{v:.1%}" for v in row] for row in heatmap_df.values],
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            hovertemplate="Region: %{y}<br>Product: %{x}<br>Return Rate: %{text}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Return Rate", tickformat=".0%"),
        ))
        fig_heat.update_layout(
            **CHART_LAYOUT,
            title=dict(text="↩️ Return Rate Heatmap (Region × Product)", font=dict(size=15)),
            xaxis=dict(title="Product", side="bottom"),
            yaxis=dict(title="Region"),
        )
        st.plotly_chart(fig_heat, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 9 – Top 10 Products (by net revenue)
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏅 Top Products & Distribution</div>',
                unsafe_allow_html=True)

    c9, c10 = st.columns(2, gap="medium")

    with c9:
        top10 = (
            df.groupby("product")["net_revenue"]
            .sum()
            .reset_index()
            .sort_values("net_revenue", ascending=True)
            .tail(10)
        )
        fig_top = go.Figure(go.Bar(
            x=top10["net_revenue"],
            y=top10["product"],
            orientation="h",
            marker=dict(
                color=top10["net_revenue"],
                colorscale=[[0, "#6C63FF"], [0.5, "#43E8D8"], [1, "#2ECC71"]],
                showscale=False,
                line=dict(width=0),
            ),
            text=[fmt_currency(v) for v in top10["net_revenue"]],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>",
        ))
        fig_top.update_layout(
            **CHART_LAYOUT,
            title=dict(text="🏅 Top Products by Revenue", font=dict(size=15)),
            xaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_top, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 10 – Sales Distribution Histogram
    # ─────────────────────────────────────────────────────────────────────────────
    with c10:
        fig_hist = go.Figure()
        for prod, color in zip(df["product"].unique(), PRODUCT_COLORS):
            subset = df[df["product"] == prod]["net_revenue"]
            fig_hist.add_trace(go.Histogram(
                x=subset,
                name=prod,
                opacity=0.7,
                nbinsx=25,
                marker_color=color,
                hovertemplate=f"<b>{prod}</b><br>Revenue: ₹%{{x:,.0f}}<extra></extra>",
            ))
        fig_hist.update_layout(
            **CHART_LAYOUT,
            barmode="overlay",
            title=dict(text="📊 Net Revenue Distribution by Product", font=dict(size=15)),
            xaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_hist, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # CHART 11 – Correlation Matrix
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔗 Correlation Matrix</div>',
                unsafe_allow_html=True)

    corr_cols = ["unit_price", "quantity_sold", "discount", "total_sales",
                 "net_revenue", "discount_amount", "shipping_cost", "returned"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_mat  = df[corr_cols].corr().round(3)

    fig_corr = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale=[
            [0.0,  "#E74C3C"],
            [0.5,  "#1A1D27"],
            [1.0,  "#2ECC71"],
        ],
        zmid=0,
        text=[[f"{v:.2f}" for v in row] for row in corr_mat.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{y} × %{x}: %{text}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    fig_corr.update_layout(
        **CHART_LAYOUT,
        title=dict(text="🔗 Feature Correlation Matrix", font=dict(size=15)),
        height=440,
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_corr, width="stretch")


    # ─────────────────────────────────────────────────────────────────────────────
    # Business Insights
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💡 Automated Business Insights</div>',
                unsafe_allow_html=True)

    insights = generate_insights(df, kpis)
    ins_cols  = st.columns(2, gap="medium")

    for i, insight in enumerate(insights):
        with ins_cols[i % 2]:
            st.markdown(f"""
            <div class="insight-card {insight['sentiment']}">
                <div class="insight-title">{insight['title']}</div>
                <div class="insight-detail">{insight['detail']}</div>
            </div>
            """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # ML Forecasting Section
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🤖 ML Sales Forecast (Linear Regression)</div>',
                unsafe_allow_html=True)

    forecast_result, future_df = get_forecast(df_full)   # always use full dataset

    monthly_ts = forecast_result["monthly"]
    train_idx  = forecast_result["train_idx"]
    test_idx   = forecast_result["test_idx"]
    y_pred_tr  = forecast_result["y_pred_train"]
    y_pred_te  = forecast_result["y_pred_test"]
    metrics    = forecast_result["metrics"]

    # ── Metrics row ──────────────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
    for col, (name, val) in zip(
        [mc1, mc2, mc3, mc4],
        [
            ("Model",  "Linear Regression"),
            ("MAE",    fmt_currency(metrics["MAE"])),
            ("RMSE",   fmt_currency(metrics["RMSE"])),
            ("R2 Score", f"{metrics['R2']:.4f}"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-name">{name}</div>
                <div class="metric-val">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast chart ───────────────────────────────────────────────────────────
    fig_fc = go.Figure()

    # Actual – training
    fig_fc.add_trace(go.Scatter(
        x=monthly_ts.loc[train_idx, "year_month"],
        y=monthly_ts.loc[train_idx, "total_revenue"],
        mode="lines+markers",
        name="Actual (Train)",
        line=dict(color="#6C63FF", width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ))

    # Predicted – training
    fig_fc.add_trace(go.Scatter(
        x=monthly_ts.loc[train_idx, "year_month"],
        y=y_pred_tr,
        mode="lines",
        name="Predicted (Train)",
        line=dict(color="#43E8D8", width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Predicted: ₹%{y:,.0f}<extra></extra>",
    ))

    # Actual + Predicted – test
    if test_idx:
        fig_fc.add_trace(go.Scatter(
            x=monthly_ts.loc[test_idx, "year_month"],
            y=monthly_ts.loc[test_idx, "total_revenue"],
            mode="lines+markers",
            name="Actual (Test)",
            line=dict(color="#F39C12", width=2),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_fc.add_trace(go.Scatter(
            x=monthly_ts.loc[test_idx, "year_month"],
            y=y_pred_te,
            mode="lines",
            name="Predicted (Test)",
            line=dict(color="#E74C3C", width=2, dash="dot"),
            hovertemplate="<b>%{x}</b><br>Predicted: ₹%{y:,.0f}<extra></extra>",
        ))

    # Future forecast
    fig_fc.add_trace(go.Scatter(
        x=future_df["year_month"],
        y=future_df["predicted_revenue"],
        mode="lines+markers",
        name="Forecast (Next 6M)",
        line=dict(color="#2ECC71", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="star", color="#2ECC71"),
        hovertemplate="<b>%{x}</b><br>Forecast: ₹%{y:,.0f}<extra></extra>",
    ))

    # Vertical separator – use add_shape (works with string x-axis in Plotly 6)
    split_month = monthly_ts["year_month"].iloc[-1]
    fig_fc.add_shape(
        type="line",
        x0=split_month, x1=split_month,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
    )
    fig_fc.add_annotation(
        x=split_month, y=1.02,
        xref="x", yref="paper",
        text="Forecast ->",
        showarrow=False,
        font=dict(color="#EAEAEA", size=11),
        xanchor="left",
    )

    fig_fc.update_layout(
        **CHART_LAYOUT,
        title=dict(text="🤖 Monthly Revenue – Actual vs Predicted vs Forecast",
                   font=dict(size=15)),
        yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(title="Month", tickangle=-45, gridcolor="rgba(255,255,255,0.04)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
    )
    st.plotly_chart(fig_fc, width="stretch")

    # ── Future table ─────────────────────────────────────────────────────────────
    with st.expander("📋 View 6-Month Forecast Data"):
        display_future = future_df.copy()
        display_future["predicted_revenue"] = display_future["predicted_revenue"].apply(
            lambda x: fmt_currency(x)
        )
        display_future.columns = ["Month", "Predicted Revenue"]
        st.dataframe(display_future, width="stretch", hide_index=True)


    # ── Hot Product Predictions (ML) ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔥 Hot Product Predictions (ML)</div>',
                unsafe_allow_html=True)
    st.markdown("We use per-product ML forecasting (Linear Regression with seasonal encoding) to predict which products will grow the most in the next 6 months compared to the last 6 months.")

    growth_df = analyze_product_growth(df, n_months=6)

    if not growth_df.empty:
        top_product = growth_df.iloc[0]["product"]
        top_growth = growth_df.iloc[0]["growth_pct"]
    
        st.info(f"**Top Hot Take**: **{top_product}** is projected to grow by **{top_growth:,.1f}%** in the next 6 months!")
    
        # Chart: Projected Growth %
        fig_hot = px.bar(
            growth_df,
            x="product",
            y="growth_pct",
            color="growth_pct",
            color_continuous_scale=[[0, "#FF6584"], [0.5, "#43E8D8"], [1, "#2ECC71"]],
            text_auto=".1f",
            title="Projected 6-Month Growth % by Product",
            labels={"growth_pct": "Projected Growth (%)", "product": "Product"},
        )
        fig_hot.update_layout(**CHART_LAYOUT, height=400)
        st.plotly_chart(fig_hot, width="stretch")
    
        with st.expander("📋 View Product Growth Data"):
            st.dataframe(
                growth_df.style.format({
                    "past_revenue": "₹{:,.0f}",
                    "predicted_revenue": "₹{:,.0f}",
                    "growth_pct": "{:.2f}%"
                }),
                width="stretch",
                hide_index=True
            )
    else:
        st.warning("Not enough data to calculate product growth trends.")

    # ─────────────────────────────────────────────────────────────────────────────
    # EDA Summary Table (Pandas describe)
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Exploratory Data Analysis Summary</div>',
                unsafe_allow_html=True)

    eda_cols = ["unit_price", "quantity_sold", "discount", "net_revenue",
                "shipping_cost", "returned"]
    eda_cols = [c for c in eda_cols if c in df.columns]
    eda_desc = df[eda_cols].describe().T.round(2)
    eda_desc.index.name = "Column"
    st.dataframe(eda_desc.reset_index(), width="stretch", hide_index=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # Pivot Table
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("📊 Revenue Pivot Table: Region × Product"):
        pivot = build_pivot_region_product(df)
        st.dataframe(
            pivot,
            width="stretch",
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # Raw Data Explorer
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("🔎 Raw Data Explorer"):
        disp_cols = ["date", "region", "product", "quantity_sold", "unit_price",
                     "discount", "net_revenue", "customer_type", "payment_method",
                     "promotion_used", "return_status"]
        disp_cols = [c for c in disp_cols if c in df.columns]
        st.dataframe(
            df[disp_cols].reset_index(drop=True),
            width="stretch",
            height=340,
        )




    # ─────────────────────────────────────────────────────────────────────────────
    # Footer (inside Overview)
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <hr>
    <div style="text-align:center; color:#7F8C8D; font-size:0.8rem; padding:10px 0 20px 0;">
        📊 Sales Analytics Dashboard · Built with <b>Streamlit</b>, <b>Plotly</b>, <b>Pandas</b>, <b>Scikit-Learn</b>
        · Data Analyst Portfolio Project
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXECUTIVE INTELLIGENCE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
with tab_exec:
    st.markdown("### 🧠 Executive Intelligence Summary")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-8px;'>Automated narrative analysis based on current filters. Refreshes with every filter change.</p>", unsafe_allow_html=True)

    summary = generate_executive_summary(df, kpis)

    # Headline banner
    trend_color = "#2ECC71" if summary["trend_pct"] >= 0 else "#FF6584"
    trend_arrow = "▲" if summary["trend_pct"] >= 0 else "▼"
    st.markdown(f"""
    <div style="background:var(--card-bg); border:1px solid var(--card-border); border-left:4px solid {trend_color};
        border-radius:12px; padding:20px; margin-bottom:20px;">
        <div style="font-size:1.05rem; color:var(--text-primary); line-height:1.7;">
            <span style="color:{trend_color}; font-weight:700; font-size:1.2rem;">{trend_arrow} {abs(summary['trend_pct']):.1f}%</span>
            &nbsp; {summary['headline']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns(3, gap="medium")

    with ec1:
        st.markdown("""<div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;
            color:#2ECC71; font-weight:700; margin-bottom:10px;">✅ Key Drivers</div>""", unsafe_allow_html=True)
        for d in summary["drivers"]:
            st.markdown(f"""<div style="background:rgba(46,204,113,0.08); border-left:3px solid #2ECC71;
                border-radius:6px; padding:8px 12px; margin-bottom:8px; font-size:0.88rem;
                color:var(--text-primary); line-height:1.5;">• {d}</div>""", unsafe_allow_html=True)

    with ec2:
        st.markdown("""<div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;
            color:#FF6584; font-weight:700; margin-bottom:10px;">⚠️ Risks</div>""", unsafe_allow_html=True)
        for r in summary["risks"]:
            st.markdown(f"""<div style="background:rgba(255,101,132,0.08); border-left:3px solid #FF6584;
                border-radius:6px; padding:8px 12px; margin-bottom:8px; font-size:0.88rem;
                color:var(--text-primary); line-height:1.5;">• {r}</div>""", unsafe_allow_html=True)

    with ec3:
        st.markdown("""<div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;
            color:#43E8D8; font-weight:700; margin-bottom:10px;">🎯 Recommended Actions</div>""", unsafe_allow_html=True)
        for a in summary["actions"]:
            st.markdown(f"""<div style="background:rgba(67,232,216,0.08); border-left:3px solid #43E8D8;
                border-radius:6px; padding:8px 12px; margin-bottom:8px; font-size:0.88rem;
                color:var(--text-primary); line-height:1.5;">→ {a}</div>""", unsafe_allow_html=True)

    # Text summary for export
    summary_txt = (
        f"EXECUTIVE SUMMARY\n{'='*50}\n\n"
        f"Trend: {trend_arrow} {abs(summary['trend_pct']):.1f}%\n"
        f"{summary['headline']}\n\n"
        f"KEY DRIVERS:\n" + "\n".join(f"  • {d}" for d in summary["drivers"]) + "\n\n"
        f"RISKS:\n"        + "\n".join(f"  • {r}" for r in summary["risks"])   + "\n\n"
        f"ACTIONS:\n"      + "\n".join(f"  → {a}" for a in summary["actions"])
    )
    st.download_button(
        "📥 Download Summary (.txt)", data=summary_txt.encode(),
        file_name="executive_summary.txt", mime="text/plain",
        use_container_width=False,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOP MOVERS INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════
with tab_movers:
    st.markdown("### 📈 Top Movers Intelligence")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-8px;'>Automatically identifies the fastest growing, declining, and standout performers in the filtered period.</p>", unsafe_allow_html=True)

    movers = get_top_movers(df)

    if movers:
        def mover_card(title, value, delta, icon, color, subtitle=""):
            arrow = "▲" if delta >= 0 else "▼"
            sign  = "+" if delta >= 0 else ""
            return f"""
            <div style="background:var(--card-bg); border:1px solid var(--card-border);
                border-top:3px solid {color}; border-radius:12px; padding:18px; height:100%;">
                <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
                <div style="font-size:0.72rem; color:var(--text-secondary); text-transform:uppercase;
                    letter-spacing:0.07em; font-weight:600; margin-bottom:6px;">{title}</div>
                <div style="font-size:1.15rem; font-weight:700; color:var(--text-primary);">{value}</div>
                <div style="font-size:0.95rem; color:{color}; font-weight:600; margin-top:4px;">
                    {arrow} {sign}{delta:.1f}%</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:4px;">{subtitle}</div>
            </div>"""

        def static_card(title, value, icon, color, subtitle=""):
            return f"""
            <div style="background:var(--card-bg); border:1px solid var(--card-border);
                border-top:3px solid {color}; border-radius:12px; padding:18px; height:100%;">
                <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
                <div style="font-size:0.72rem; color:var(--text-secondary); text-transform:uppercase;
                    letter-spacing:0.07em; font-weight:600; margin-bottom:6px;">{title}</div>
                <div style="font-size:1.15rem; font-weight:700; color:var(--text-primary);">{value}</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:6px;">{subtitle}</div>
            </div>"""

        mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
        with mc1:
            st.markdown(mover_card("Top Growing Product", movers["top_growing_product"],
                movers["top_growing_pct"], "🚀", "#2ECC71", "vs. prior period"), unsafe_allow_html=True)
        with mc2:
            st.markdown(mover_card("Top Declining Product", movers["top_declining_product"],
                movers["top_declining_pct"], "📉", "#FF6584", "vs. prior period"), unsafe_allow_html=True)
        with mc3:
            st.markdown(static_card("🏆 Top Revenue Product", movers["top_revenue_product"],
                "⭐", "#6C63FF", "Highest total revenue in period"), unsafe_allow_html=True)
        with mc4:
            st.markdown(static_card("Best Promotion", movers["best_promo"],
                "🎁", "#43E8D8", f"Avg order ₹{movers['best_promo_aov']:,.0f}"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        mc5, mc6, mc7, mc8 = st.columns(4, gap="medium")
        with mc5:
            st.markdown(static_card("Best Region", movers["top_revenue_region"],
                "🏆", "#2ECC71", "Highest revenue region"), unsafe_allow_html=True)
        with mc6:
            st.markdown(static_card("⚠️ Weakest Region", movers["worst_revenue_region"],
                "🗺️", "#FF6584", "Lowest revenue region"), unsafe_allow_html=True)
        with mc7:
            st.markdown(static_card("Highest Returns Product", movers["highest_return_product"],
                "↩️", "#F39C12",
                f"{movers['highest_return_rate']:.1f}% return rate"), unsafe_allow_html=True)
        with mc8:
            st.markdown(static_card("Top Revenue Region", movers["top_revenue_region"],
                "📍", "#6C63FF", "By total net revenue"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Product movers chart
        if not movers["product_movers"].empty:
            pm_df = movers["product_movers"].copy()
            pm_df["color"] = pm_df["delta_pct"].apply(lambda x: "#2ECC71" if x >= 0 else "#FF6584")
            fig_pm = px.bar(
                pm_df, x="product", y="delta_pct", color="delta_pct",
                color_continuous_scale=[[0, "#FF6584"], [0.5, "#F39C12"], [1, "#2ECC71"]],
                text_auto=".1f",
                labels={"delta_pct": "Period-over-Period Change (%)", "product": "Product"},
            )
            fig_pm.update_layout(**CHART_LAYOUT, title="📊 Product Revenue Change: First Half vs. Second Half of Period",
                                  coloraxis_showscale=False)
            fig_pm.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            st.plotly_chart(fig_pm, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — BUSINESS EXPLORER (Replacing Drill-Down & Compare)
# ═════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    render_business_explorer(df)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — WHAT-IF SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown("### 🎛️ Business Scenario Simulator (What-If Analysis)")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-8px;'>Adjust business levers and see projected revenue impact instantly. Based on a multiplicative elasticity model applied to current filtered data.</p>", unsafe_allow_html=True)

    ws1, ws2 = st.columns([2, 3], gap="large")

    with ws1:
        st.markdown("#### ⚙️ Adjust Levers")
        discount_delta = st.slider("Discount Rate Change (%)", -10.0, 20.0, 0.0, 0.5,
            help="Positive = higher discount. Each +1% discount reduces margin by ~1.5%")
        demand_delta   = st.slider("Demand / Volume Change (%)", -30.0, 50.0, 0.0, 1.0,
            help="Expected increase or decrease in number of orders")
        return_delta   = st.slider("Return Rate Change (%)", -5.0, 15.0, 0.0, 0.5,
            help="Positive = more returns. Each +1% return rate reduces revenue by ~0.8%")

        wi = run_what_if(df, discount_delta, demand_delta, return_delta)

    with ws2:
        if wi:
            st.markdown("#### 📊 Projected Impact")
            impact_color = "#2ECC71" if wi["delta_pct"] >= 0 else "#FF6584"
            impact_arrow = "▲" if wi["delta_pct"] >= 0 else "▼"

            # Hero card
            st.markdown(f"""
            <div style="background:var(--card-bg); border:1px solid var(--card-border);
                border-left:4px solid {impact_color}; border-radius:12px; padding:20px; margin-bottom:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:0.75rem; color:var(--text-secondary); font-weight:600; margin-bottom:4px;">CURRENT REVENUE</div>
                        <div style="font-size:1.6rem; font-weight:800; color:var(--text-primary);">{fmt_currency(wi['base_revenue'])}</div>
                    </div>
                    <div style="font-size:2rem; color:{impact_color}; font-weight:800;">{impact_arrow} {abs(wi['delta_pct']):.1f}%</div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem; color:var(--text-secondary); font-weight:600; margin-bottom:4px;">PROJECTED REVENUE</div>
                        <div style="font-size:1.6rem; font-weight:800; color:{impact_color};">{fmt_currency(wi['proj_revenue'])}</div>
                    </div>
                </div>
                <div style="text-align:center; margin-top:8px; font-size:0.9rem; color:{impact_color}; font-weight:600;">
                    Delta: {"+" if wi['delta_revenue'] >= 0 else ""}{fmt_currency(wi['delta_revenue'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            wm1, wm2, wm3, wm4 = st.columns(4, gap="small")
            wm1.metric("Orders (Current)",   f"{wi['base_orders']:,.0f}")
            wm2.metric("Orders (Projected)", f"{wi['proj_orders']:,.0f}",
                       delta=f"{wi['proj_orders']-wi['base_orders']:+.0f}")
            wm3.metric("AOV (Current)",   fmt_currency(wi["base_aov"]))
            wm4.metric("AOV (Projected)", fmt_currency(wi["proj_aov"]),
                       delta=fmt_currency(wi["proj_aov"] - wi["base_aov"]))

            # Sensitivity waterfall
            components = pd.DataFrame([
                {"Component": "Base Revenue",     "Value":  wi["base_revenue"],  "type": "base"},
                {"Component": "Demand Effect",    "Value":  wi["base_revenue"] * (demand_delta/100), "type": "positive" if demand_delta >= 0 else "negative"},
                {"Component": "Discount Effect",  "Value":  wi["base_revenue"] * (-discount_delta/100) * 1.5, "type": "negative" if discount_delta > 0 else "positive"},
                {"Component": "Return Effect",    "Value":  wi["base_revenue"] * (-return_delta/100) * 0.8, "type": "negative" if return_delta > 0 else "positive"},
                {"Component": "Projected Revenue","Value":  wi["proj_revenue"], "type": "total"},
            ])
            colors = {"base": "#6C63FF", "positive": "#2ECC71", "negative": "#FF6584", "total": "#43E8D8"}
            fig_wf = go.Figure(go.Bar(
                x=components["Component"],
                y=components["Value"],
                marker_color=[colors[t] for t in components["type"]],
                text=[fmt_currency(v) for v in components["Value"]],
                textposition="auto",
            ))
            fig_wf.update_layout(**CHART_LAYOUT, title="💡 Revenue Sensitivity Breakdown",
                                  showlegend=False,
                                  yaxis=dict(title="Revenue Impact (₹)", gridcolor="rgba(255,255,255,0.06)"))
            st.plotly_chart(fig_wf, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — ANOMALY DETECTION CENTER
# ═════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown("### ⚠️ Anomaly Detection Center")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-8px;'>Statistical anomaly detection using Z-Score thresholds and Isolation Forest on monthly aggregates.</p>", unsafe_allow_html=True)

    anom_result = detect_anomalies(df)
    monthly_df  = anom_result.get("monthly_df", pd.DataFrame())
    anomalies   = anom_result.get("anomalies", [])

    if monthly_df.empty:
        st.warning("Not enough monthly data to run anomaly detection (need ≥ 4 months).")
    else:
        # Summary badges
        n_anom = len(anomalies)
        anom_label = "anomalies" if n_anom != 1 else "anomaly"
        st.markdown(f"""<div style="background:{'rgba(255,101,132,0.12)' if n_anom > 0 else 'rgba(46,204,113,0.12)'};
            border:1px solid {'#FF6584' if n_anom > 0 else '#2ECC71'};
            border-radius:10px; padding:12px 18px; margin-bottom:16px; font-size:0.95rem;
            color:var(--text-primary);">
            {'⚠️' if n_anom > 0 else '✅'} <b>{n_anom} {anom_label}</b> detected across {len(monthly_df)} months
            (Z-score threshold: |z| > 2.0 · Isolation Forest contamination: 15%)
        </div>""", unsafe_allow_html=True)

        # Time series with anomaly dots
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=monthly_df["year_month"], y=monthly_df["revenue"],
            mode="lines+markers", name="Monthly Revenue",
            line=dict(color="#6C63FF", width=2),
            marker=dict(size=5, color="#6C63FF"),
            hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
        ))
        if "is_anomaly" in monthly_df.columns:
            anom_pts = monthly_df[monthly_df["is_anomaly"]]
            if not anom_pts.empty:
                fig_anom.add_trace(go.Scatter(
                    x=anom_pts["year_month"], y=anom_pts["revenue"],
                    mode="markers", name="Anomaly",
                    marker=dict(size=14, color="#FF6584", symbol="x",
                                line=dict(width=2, color="#FF6584")),
                    hovertemplate="<b>⚠️ ANOMALY: %{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
                ))
        fig_anom.update_layout(**CHART_LAYOUT,
            title="📅 Monthly Revenue with Anomalies Marked",
            yaxis=dict(title="Net Revenue (₹)", gridcolor="rgba(255,255,255,0.06)"))
        st.plotly_chart(fig_anom, use_container_width=True)

        # Anomaly detail cards
        if anomalies:
            st.markdown("#### 🔎 Detected Anomalies")
            for anom in anomalies:
                sev_color = "#FF6584" if "High" in anom["severity"] else "#F39C12"
                reasons_html = " &nbsp;·&nbsp; ".join(anom["reasons"])
                st.markdown(f"""
                <div style="background:var(--card-bg); border:1px solid {sev_color};
                    border-left:4px solid {sev_color}; border-radius:10px;
                    padding:14px 18px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; font-size:1rem; color:var(--text-primary);">{anom['month']}</span>
                            &nbsp; <span style="font-size:0.8rem; color:{sev_color};">{anom['severity']}</span>
                        </div>
                        <div style="font-size:0.9rem; color:var(--text-primary);">
                            Revenue: <b>{fmt_currency(anom['revenue'])}</b> &nbsp;|&nbsp;
                            Orders: <b>{anom['orders']:,.0f}</b> &nbsp;|&nbsp;
                            Return Rate: <b>{anom['ret_rate']:.1f}%</b>
                        </div>
                    </div>
                    <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:8px;">
                        {reasons_html}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No significant anomalies detected in the current filtered data.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — EXPORT CENTER
# ═════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("### 📤 Export Center")
    st.markdown("<p style='color:var(--text-secondary); margin-top:-8px;'>Download your filtered data, summaries, and reports in multiple formats.</p>", unsafe_allow_html=True)

    ex1, ex2, ex3 = st.columns(3, gap="medium")

    with ex1:
        st.markdown("""<div style="background:var(--card-bg); border:1px solid var(--card-border);
            border-radius:12px; padding:20px; text-align:center; margin-bottom:12px;">
            <div style="font-size:2.5rem;">📊</div>
            <h4 style="margin:8px 0 4px 0; color:var(--text-primary);">Excel Report</h4>
            <p style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:12px;">
            4 sheets: KPIs · Products · Regions · Raw Data (up to 5,000 rows)</p>
        </div>""", unsafe_allow_html=True)
        excel_bytes = build_excel_export(df, kpis, st.session_state.applied_filters)
        st.download_button(
            "📥 Download Excel (.xlsx)", data=excel_bytes,
            file_name="sales_dashboard_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary",
        )

    with ex2:
        st.markdown("""<div style="background:var(--card-bg); border:1px solid var(--card-border);
            border-radius:12px; padding:20px; text-align:center; margin-bottom:12px;">
            <div style="font-size:2.5rem;">📝</div>
            <h4 style="margin:8px 0 4px 0; color:var(--text-primary);">Executive Summary</h4>
            <p style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:12px;">
            Plain-text narrative summary with drivers, risks, and recommended actions</p>
        </div>""", unsafe_allow_html=True)
        summary_ex = generate_executive_summary(df, kpis)
        txt_lines = (
            "SALES ANALYTICS — EXECUTIVE SUMMARY\n"
            f"Generated from filtered dataset ({len(df):,} rows)\n"
            f"Filters: {st.session_state.applied_filters}\n"
            "=" * 55 + "\n\n"
            f"HEADLINE:\n{summary_ex['headline']}\n\n"
            "KEY DRIVERS:\n" + "\n".join(f"  • {d}" for d in summary_ex["drivers"]) + "\n\n"
            "RISKS:\n"        + "\n".join(f"  • {r}" for r in summary_ex["risks"])   + "\n\n"
            "RECOMMENDED ACTIONS:\n" + "\n".join(f"  → {a}" for a in summary_ex["actions"])
        )
        st.download_button(
            "📥 Download Summary (.txt)", data=txt_lines.encode(),
            file_name="executive_summary.txt", mime="text/plain",
            use_container_width=True,
        )

    with ex3:
        st.markdown("""<div style="background:var(--card-bg); border:1px solid var(--card-border);
            border-radius:12px; padding:20px; text-align:center; margin-bottom:12px;">
            <div style="font-size:2.5rem;">📋</div>
            <h4 style="margin:8px 0 4px 0; color:var(--text-primary);">Filtered CSV</h4>
            <p style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:12px;">
            Export the full filtered dataset as a CSV file for further analysis</p>
        </div>""", unsafe_allow_html=True)
        csv_cols = [c for c in ["date","region","product","quantity_sold","unit_price",
                    "discount","net_revenue","customer_type","payment_method",
                    "promotion_used","returned"] if c in df.columns]
        csv_bytes = df[csv_cols].to_csv(index=False).encode()
        st.download_button(
            "📥 Download CSV", data=csv_bytes,
            file_name="filtered_sales_data.csv", mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.2);
        border-radius:10px; padding:14px 18px; font-size:0.82rem; color:var(--text-secondary);">
        <b>📌 Export Notes:</b> All exports respect your current sidebar filters and date range.
        The Excel file contains formatted sheets with auto-sized columns.
        Full PDF export (with embedded charts) requires a server-side PDF engine — contact your administrator.
    </div>""", unsafe_allow_html=True)
