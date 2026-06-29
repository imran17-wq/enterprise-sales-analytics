import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.utils import get_chart_layout, fmt_currency, fmt_number, fmt_pct

METRIC_MAPPING = {
    "Revenue": ("net_revenue", "sum", "currency"),
    "Orders": ("order_id", "nunique", "number"),
    "Quantity Sold": ("quantity_sold", "sum", "number"),
    "Average Order Value": ("net_revenue", "mean", "currency"),
    "Return Rate": ("returned", "mean", "percent"),
    "Profit": ("profit", "sum", "currency")
}

DIM_MAPPING = {
    "Region": "region",
    "Product": "product",
    "Customer Type": "customer_type",
    "Payment Method": "payment_method",
    "Promotion": "promotion_used",
    "Month": "month",
}

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist
    df_explorer = df.copy()
    if "profit" not in df_explorer.columns and "net_revenue" in df_explorer.columns:
        df_explorer["profit"] = df_explorer["net_revenue"] * 0.2
    if "month" not in df_explorer.columns and "date" in df_explorer.columns:
        df_explorer["month"] = pd.to_datetime(df_explorer["date"]).dt.strftime("%Y-%m")
    if "customer_type" not in df_explorer.columns:
        # mock customer type if missing
        import numpy as np
        np.random.seed(42)
        df_explorer["customer_type"] = np.random.choice(["Retail", "Wholesale", "Enterprise"], size=len(df_explorer))
    return df_explorer

def auto_generate_chart(df: pd.DataFrame, dim1: str, dim2: str, metric: str, layout: dict):
    col, agg_func, fmt = METRIC_MAPPING[metric]
    
    if not dim2 or dim2 == "None":
        # 1 Dimension
        agg_df = df.groupby(dim1)[col].agg(agg_func).reset_index().sort_values(by=col, ascending=False)
        
        # Decide chart type
        if dim1 == "month":
            # Line chart for time
            agg_df = agg_df.sort_values(by="month")
            fig = px.line(agg_df, x=dim1, y=col, markers=True, title=f"{metric} over Time")
            fig.update_traces(line=dict(width=3, color=layout["colorway"][0]), fill='tozeroy', fillcolor=f"rgba(129,140,248,0.1)")
            reco = "Line Chart"
        elif len(agg_df) <= 5 and dim1 != "month":
            # Donut for few categories
            fig = px.pie(agg_df, names=dim1, values=col, hole=0.6, title=f"{metric} Composition by {dim1.title()}")
            fig.update_traces(textinfo='percent+label', marker=dict(colors=layout["colorway"]))
            reco = "Donut Chart"
        else:
            # Bar chart default
            fig = px.bar(agg_df, x=dim1, y=col, title=f"{metric} by {dim1.title()}", color=dim1, color_discrete_sequence=layout["colorway"])
            reco = "Bar Chart"

    else:
        # 2 Dimensions
        if dim1 == dim2:
            st.warning("Please select distinct dimensions.")
            return None, "None"
        
        agg_df = df.groupby([dim1, dim2])[col].agg(agg_func).reset_index()
        
        if dim1 == "month" or dim2 == "month":
            time_col = "month"
            cat_col = dim2 if dim1 == "month" else dim1
            agg_df = agg_df.sort_values(by=time_col)
            fig = px.area(agg_df, x=time_col, y=col, color=cat_col, title=f"{metric} over Time by {cat_col.title()}", color_discrete_sequence=layout["colorway"])
            reco = "Stacked Area Chart"
        elif df[dim1].nunique() < 10 and df[dim2].nunique() < 10:
            # Grouped bar chart
            fig = px.bar(agg_df, x=dim1, y=col, color=dim2, barmode='group', title=f"{metric} by {dim1.title()} & {dim2.title()}", color_discrete_sequence=layout["colorway"])
            reco = "Grouped Bar Chart"
        else:
            # Hierarchical Sunburst
            agg_df[col] = agg_df[col].apply(lambda x: max(x, 0)) # Sunburst needs positive values
            fig = px.sunburst(agg_df, path=[dim1, dim2], values=col, title=f"{metric} Hierarchy: {dim1.title()} → {dim2.title()}", color_discrete_sequence=layout["colorway"])
            reco = "Sunburst Chart"

    fig.update_layout(**layout)
    return fig, reco

def generate_pivot_table(df: pd.DataFrame, dim1: str, dim2: str, metric: str):
    col, agg_func, fmt = METRIC_MAPPING[metric]
    
    if not dim2 or dim2 == "None":
        pivot = df.groupby(dim1)[col].agg(agg_func).reset_index().sort_values(by=col, ascending=False)
        pivot.columns = [dim1.title(), metric]
    else:
        pivot = df.pivot_table(index=dim1, columns=dim2, values=col, aggfunc=agg_func, fill_value=0)
        # Add row totals for sorting
        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values(by="Total", ascending=False).drop(columns=["Total"])
    
    # Apply styling
    if fmt == "currency":
        formatter = lambda x: f"₹{x:,.0f}" if isinstance(x, (int, float)) else x
    elif fmt == "percent":
        formatter = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
    else:
        formatter = lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x

    return pivot.style.format(formatter).background_gradient(cmap='Blues', axis=None)

def render_compare_mode(df: pd.DataFrame, dim: str, a: str, b: str, metric: str, layout: dict):
    col, agg_func, fmt = METRIC_MAPPING[metric]
    
    df_a = df[df[dim] == a]
    df_b = df[df[dim] == b]
    
    val_a = df_a[col].agg(agg_func)
    val_b = df_b[col].agg(agg_func)
    
    # Avoid div/0
    diff = val_a - val_b
    pct_diff = (diff / val_b * 100) if val_b and val_b != 0 else 0
    
    # Formatter
    if fmt == "currency":
        str_a, str_b, str_diff = fmt_currency(val_a), fmt_currency(val_b), fmt_currency(abs(diff))
    elif fmt == "percent":
        str_a, str_b, str_diff = fmt_pct(val_a), fmt_pct(val_b), fmt_pct(abs(diff))
    else:
        str_a, str_b, str_diff = fmt_number(val_a), fmt_number(val_b), fmt_number(abs(diff))

    # Layout UI
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="insight-card"><b>{a}</b><br><span style="font-size:2rem; font-weight:700;">{str_a}</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="insight-card"><b>{b}</b><br><span style="font-size:2rem; font-weight:700;">{str_b}</span></div>', unsafe_allow_html=True)
    with c3:
        color = "positive" if diff > 0 else "negative" if diff < 0 else "neutral"
        sign = "+" if diff > 0 else "-" if diff < 0 else ""
        st.markdown(f'<div class="insight-card {color}"><b>Difference (A vs B)</b><br><span style="font-size:2rem; font-weight:700;">{sign}{str_diff}</span><br>{sign}{pct_diff:.1f}%</div>', unsafe_allow_html=True)
        
    # AI Insight text
    winner = a if diff > 0 else b
    loser = b if diff > 0 else a
    st.markdown("### 🤖 AI Insights")
    st.info(f"**{winner}** outperforms **{loser}** in {metric} by **{abs(pct_diff):.1f}%** ({str_diff}). "
            f"Consider exploring a secondary dimension (like Product or Month) to identify the specific segments driving this difference.")

    # Side by side chart breakdown by secondary dimension (e.g. Month)
    st.markdown("### Breakdown over Time")
    agg_time = df[df[dim].isin([a, b])].groupby([dim, "month"])[col].agg(agg_func).reset_index()
    if not agg_time.empty:
        fig = px.bar(agg_time, x="month", y=col, color=dim, barmode='group', title=f"{metric} Comparison over Time", color_discrete_sequence=layout["colorway"])
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

def render_business_explorer(df: pd.DataFrame):
    df_exp = prep_df(df)
    layout = get_chart_layout(st.session_state.theme)

    # Initialize Drill Path
    if "explorer_path" not in st.session_state:
        st.session_state.explorer_path = []

    st.markdown("""
    <div style="background: var(--card-bg); padding: 16px; border-radius: 12px; margin-bottom: 24px; border: 1px solid var(--card-border);">
        <h3 style="margin-top: 0;">🔭 Interactive Business Explorer</h3>
        <p style="color: var(--text-secondary); margin-bottom: 0;">Dynamically generate charts, drill into dimensions, and perform entity comparisons without static reports.</p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Exploration Mode", ["📈 Pivot & Auto-Chart", "⚖️ Compare Entities"], horizontal=True)

    if mode == "📈 Pivot & Auto-Chart":
        col1, col2, col3 = st.columns(3)
        with col1:
            dim1_label = st.selectbox("Primary Dimension", list(DIM_MAPPING.keys()), index=0)
        with col2:
            dim2_label = st.selectbox("Secondary Dimension (Optional)", ["None"] + list(DIM_MAPPING.keys()), index=0)
        with col3:
            metric_label = st.selectbox("Metric to Analyze", list(METRIC_MAPPING.keys()), index=0)

        dim1 = DIM_MAPPING[dim1_label]
        dim2 = DIM_MAPPING.get(dim2_label, "None")

        # Drill Down Context
        active_df = df_exp.copy()
        if st.session_state.explorer_path:
            # We filter by the FIRST path element assuming it corresponds to Dim1 for simplicity
            # In a full app, path would store (dim, val) pairs
            st.markdown(f"**🔍 Active Drill Filter:** `{' > '.join(st.session_state.explorer_path)}`")
            if st.button("❌ Clear Drill-Down"):
                st.session_state.explorer_path = []
                st.rerun()
            
            # Simple apply logic: just filter dim1 by path[0] if valid
            # For robustness in this demo, we'll just filter `region` or `product` globally if they match.
            for filter_val in st.session_state.explorer_path:
                # search all columns for the filter_val and apply
                matched_cols = [c for c in active_df.columns if active_df[c].astype(str).eq(filter_val).any()]
                if matched_cols:
                    active_df = active_df[active_df[matched_cols[0]] == filter_val]

        if active_df.empty:
            st.warning("No data available for current drill-down selection.")
            return

        fig, reco = auto_generate_chart(active_df, dim1, dim2, metric_label, layout)
        
        st.markdown(f"**🏆 Recommended Chart:** `{reco}`")
        if fig:
            # Display chart
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
            
            # Catch drill down clicks
            if event and event.get("selection", {}).get("points"):
                pts = event["selection"]["points"]
                if pts:
                    # depending on chart, x or label holds the dimension
                    click_val = pts[0].get("x", pts[0].get("label"))
                    if click_val and click_val not in st.session_state.explorer_path:
                        st.session_state.explorer_path.append(click_val)
                        st.rerun()

        st.markdown("### 🧮 Dynamic Pivot Table")
        pivot = generate_pivot_table(active_df, dim1, dim2, metric_label)
        st.dataframe(pivot, use_container_width=True)

    else:
        # COMPARE MODE
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dim_label = st.selectbox("Compare Dimension", list(DIM_MAPPING.keys()), index=0)
            dim = DIM_MAPPING[dim_label]
        
        unique_vals = df_exp[dim].dropna().unique().tolist()
        if len(unique_vals) < 2:
            st.warning(f"Not enough distinct values in {dim_label} to compare.")
            return

        with col2:
            val_a = st.selectbox("Entity A", unique_vals, index=0)
        with col3:
            val_b = st.selectbox("Entity B", unique_vals, index=1 if len(unique_vals) > 1 else 0)
        with col4:
            metric_label = st.selectbox("Compare Metric", list(METRIC_MAPPING.keys()), index=0)

        if val_a == val_b:
            st.warning("Please select two different entities to compare.")
        else:
            render_compare_mode(df_exp, dim, val_a, val_b, metric_label, layout)
