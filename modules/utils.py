
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (consistent across all Plotly charts)
# ---------------------------------------------------------------------------

PALETTE = {
    "primary":      "#6C63FF",   # indigo violet
    "secondary":    "#FF6584",   # coral pink
    "accent":       "#43E8D8",   # cyan mint
    "success":      "#2ECC71",
    "warning":      "#F39C12",
    "danger":       "#E74C3C",
    "bg":           "#0E1117",   # Streamlit dark bg
    "card_bg":      "#1A1D27",
    "text":         "#EAEAEA",
    "muted":        "#7F8C8D",
}

REGION_COLORS = {
    "East":    "#6C63FF",
    "West":    "#FF6584",
    "North":   "#43E8D8",
    "South":   "#F39C12",
    "Central": "#2ECC71",
}

PRODUCT_COLORS = [
    "#818CF8", "#06B6D4", "#10B981",
    "#F43F5E", "#F59E0B", "#D946EF", "#6366F1",
]

def get_chart_layout(theme: str = "dark") -> dict:
    if theme == "light":
        return dict(
            template        = "plotly_white",
            paper_bgcolor   = "rgba(0, 0, 0, 0)",
            plot_bgcolor    = "rgba(0, 0, 0, 0)",
            font            = dict(family="Inter, -apple-system, sans-serif", color="#111827"),
            margin          = dict(l=20, r=20, t=40, b=20),
            colorway        = PRODUCT_COLORS,
        )
    else:
        return dict(
            template        = "plotly_dark",
            paper_bgcolor   = "rgba(0, 0, 0, 0)",
            plot_bgcolor    = "rgba(0, 0, 0, 0)",
            font            = dict(family="Inter, -apple-system, sans-serif", color="#F9FAFB"),
            margin          = dict(l=20, r=20, t=40, b=20),
            colorway        = PRODUCT_COLORS,
        )


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute all headline KPIs from the feature-engineered DataFrame,
    including period-over-period (PoP) deltas.
    """
    if df.empty:
        return {}

    # Basic metrics
    total_revenue   = df["net_revenue"].sum()
    total_orders    = df["order_id"].nunique()
    total_quantity  = df["quantity_sold"].sum()
    avg_order_value = df["net_revenue"].mean()
    return_rate     = df["returned"].mean() * 100
    total_profit    = df.get("profit", df["net_revenue"] * 0.2).sum() # Mock profit if not present

    # Categorical bests
    region_revenue  = df.groupby("region")["net_revenue"].sum()
    best_region     = region_revenue.idxmax() if not region_revenue.empty else "N/A"
    worst_region    = region_revenue.idxmin() if not region_revenue.empty else "N/A"

    product_revenue = df.groupby("product")["net_revenue"].sum()
    best_product    = product_revenue.idxmax() if not product_revenue.empty else "N/A"
    worst_product   = product_revenue.idxmin() if not product_revenue.empty else "N/A"

    promo_impact = df.groupby("promotion_used")["net_revenue"].mean()
    best_promo = promo_impact.idxmax() if not promo_impact.empty else "N/A"

    # Delta Calculations (First half vs Second half of the period)
    df_sorted = df.sort_values("date")
    if len(df_sorted) > 2:
        mid_idx = len(df_sorted) // 2
        df_p1 = df_sorted.iloc[:mid_idx]
        df_p2 = df_sorted.iloc[mid_idx:]

        r1 = df_p1["net_revenue"].sum()
        r2 = df_p2["net_revenue"].sum()
        rev_delta = ((r2 - r1) / r1 * 100) if r1 > 0 else 0

        o1 = df_p1["order_id"].nunique()
        o2 = df_p2["order_id"].nunique()
        ord_delta = ((o2 - o1) / o1 * 100) if o1 > 0 else 0

        q1 = df_p1["quantity_sold"].sum()
        q2 = df_p2["quantity_sold"].sum()
        qty_delta = ((q2 - q1) / q1 * 100) if q1 > 0 else 0

        a1 = df_p1["net_revenue"].mean()
        a2 = df_p2["net_revenue"].mean()
        aov_delta = ((a2 - a1) / a1 * 100) if a1 > 0 else 0

        rr1 = df_p1["returned"].mean() * 100
        rr2 = df_p2["returned"].mean() * 100
        rr_delta = rr2 - rr1 # absolute delta for percentages

        p1 = df_p1.get("profit", df_p1["net_revenue"] * 0.2).sum()
        p2 = df_p2.get("profit", df_p2["net_revenue"] * 0.2).sum()
        profit_delta = ((p2 - p1) / p1 * 100) if p1 > 0 else 0
    else:
        rev_delta = ord_delta = qty_delta = aov_delta = rr_delta = profit_delta = 0.0

    # Composite Scores
    # Revenue Health: base 80 + up to 20 points from rev growth
    rev_health = min(100, max(0, 80 + (rev_delta * 2)))
    # Growth Score: combination of orders and aov growth
    growth_score = min(100, max(0, 75 + ord_delta + aov_delta))

    return {
        "total_revenue":   total_revenue,
        "revenue_delta":   rev_delta,
        "total_profit":    total_profit,
        "profit_delta":    profit_delta,
        "total_orders":    total_orders,
        "orders_delta":    ord_delta,
        "total_quantity":  total_quantity,
        "qty_delta":       qty_delta,
        "avg_order_value": avg_order_value,
        "aov_delta":       aov_delta,
        "return_rate":     return_rate,
        "rr_delta":        rr_delta,
        "best_region":     best_region,
        "worst_region":    worst_region,
        "best_product":    best_product,
        "worst_product":   worst_product,
        "best_promo":      best_promo,
        "health_score":    rev_health,
        "growth_score":    growth_score,
    }


# ---------------------------------------------------------------------------
# Business Insights
# ---------------------------------------------------------------------------

def generate_insights(df: pd.DataFrame, kpis: dict) -> list[dict]:
    """
    Generate automated, data-driven business insights.

    Returns
    -------
    list of dicts, each with keys:
      title    : str
      detail   : str
      icon     : str (emoji)
      sentiment: 'positive' | 'negative' | 'neutral'
    """
    insights = []

    # 1. Best region
    region_rev  = df.groupby("region")["net_revenue"].sum().sort_values(ascending=False)
    best_r, best_r_val   = region_rev.index[0], region_rev.iloc[0]
    worst_r, worst_r_val = region_rev.index[-1], region_rev.iloc[-1]

    insights.append({
        "title":     "🏆 Best Performing Region",
        "detail":    (f"**{best_r}** leads with total revenue of "
                      f"**₹{best_r_val:,.0f}**, contributing "
                      f"**{best_r_val / region_rev.sum() * 100:.1f}%** of total sales."),
        "icon":      "📈",
        "sentiment": "positive",
    })

    insights.append({
        "title":     "⚠️ Lowest Performing Region",
        "detail":    (f"**{worst_r}** has the lowest revenue of "
                      f"**₹{worst_r_val:,.0f}** — "
                      f"consider targeted campaigns or resource reallocation."),
        "icon":      "📉",
        "sentiment": "negative",
    })

    # 2. Best product
    prod_rev  = df.groupby("product")["net_revenue"].sum().sort_values(ascending=False)
    best_p    = prod_rev.index[0]
    best_p_val = prod_rev.iloc[0]
    insights.append({
        "title":     "🥇 Highest Revenue Product",
        "detail":    (f"**{best_p}** is the top-selling product with "
                      f"**₹{best_p_val:,.0f}** in total revenue."),
        "icon":      "💰",
        "sentiment": "positive",
    })

    # 3. Promotion impact
    promo_avg = df.groupby("promotion_used")["net_revenue"].mean()
    no_promo_avg  = promo_avg.get("NONE", 0)
    best_promo    = promo_avg.drop("NONE", errors="ignore")
    if not best_promo.empty:
        best_p_name   = best_promo.idxmax()
        best_p_avg    = best_promo.max()
        lift          = (best_p_avg - no_promo_avg) / max(no_promo_avg, 1) * 100
        insights.append({
            "title":     "🎯 Promotion Effectiveness",
            "detail":    (f"Promotion **{best_p_name}** achieves the highest "
                          f"average order value of **₹{best_p_avg:,.0f}**, "
                          f"a **{lift:.1f}%** lift over no-promotion orders."),
            "icon":      "🎁",
            "sentiment": "positive" if lift > 0 else "neutral",
        })

    # 4. Return trends
    rr = kpis["return_rate"]
    returned_by_product = (
        df.groupby("product")["returned"].mean() * 100
    ).sort_values(ascending=False)
    highest_rr_prod = returned_by_product.index[0]

    sentiment = "negative" if rr > 15 else "positive" if rr < 8 else "neutral"
    insights.append({
        "title":     "🔄 Return Trends",
        "detail":    (f"Overall return rate is **{rr:.1f}%**. "
                      f"**{highest_rr_prod}** has the highest return rate "
                      f"(**{returned_by_product.iloc[0]:.1f}%**). "
                      + ("Action required to investigate product quality." if rr > 15
                         else "Returns are within acceptable limits.")),
        "icon":      "↩️",
        "sentiment": sentiment,
    })

    # 5. Customer purchasing patterns
    cust_rev = df.groupby("customer_type")["net_revenue"].agg(["mean", "sum"])
    if len(cust_rev) >= 2:
        top_cust = cust_rev["sum"].idxmax()
        top_aov  = cust_rev["mean"].idxmax()
        insights.append({
            "title":     "👥 Customer Purchasing Patterns",
            "detail":    (f"**{top_cust}** customers generate the most total revenue. "
                          f"**{top_aov}** customers have the highest average order value "
                          f"(**₹{cust_rev.loc[top_aov, 'mean']:,.0f}**). "
                          f"Focus loyalty programs on **{top_aov}** segment."),
            "icon":      "🛍️",
            "sentiment": "neutral",
        })

    # 6. Discount effectiveness
    discount_corr = df[["discount", "net_revenue"]].corr().iloc[0, 1]
    insights.append({
        "title":     "💸 Discount vs Revenue Correlation",
        "detail":    (f"Correlation between discount rate and net revenue: "
                      f"**{discount_corr:.3f}**. "
                      + ("Higher discounts correlate positively with revenue."
                         if discount_corr > 0
                         else "Higher discounts tend to reduce net revenue — "
                              "revisit discounting strategy.")),
        "icon":      "🔗",
        "sentiment": "positive" if discount_corr > 0 else "negative",
    })

    # 7. Monthly growth
    monthly_rev = df.groupby("year_month")["net_revenue"].sum().sort_index()
    if len(monthly_rev) >= 2:
        latest_growth = (monthly_rev.iloc[-1] - monthly_rev.iloc[-2]) / max(monthly_rev.iloc[-2], 1) * 100
        insights.append({
            "title":     "📅 Latest Monthly Growth",
            "detail":    (f"Revenue in **{monthly_rev.index[-1]}** "
                          + (f"grew by **{latest_growth:.1f}%**"
                             if latest_growth >= 0
                             else f"declined by **{abs(latest_growth):.1f}%**")
                          + f" vs the prior month."),
            "icon":      "📊",
            "sentiment": "positive" if latest_growth >= 0 else "negative",
        })

    return insights


def generate_root_cause_analysis(df: pd.DataFrame) -> str:
    """
    Simulates an AI Analyst providing deterministic root cause analysis
    for the current filtered data state.
    """
    if df.empty:
        return "Not enough data for root cause analysis."
        
    revenue = df["net_revenue"].sum()
    
    # 1. Product Analysis
    prod_rev = df.groupby("product")["net_revenue"].sum()
    top_prod = prod_rev.idxmax()
    top_prod_val = prod_rev.max()
    top_prod_pct = top_prod_val / revenue if revenue > 0 else 0
    
    # 2. Regional Differences
    reg_rev = df.groupby("region")["net_revenue"].sum()
    top_reg = reg_rev.idxmax()
    bot_reg = reg_rev.idxmin()
    
    # 3. Returns Analysis
    ret_rate = df["returned"].mean()
    high_ret_prod = df.groupby("product")["returned"].mean().idxmax()
    
    # Construct Story
    story = f"""
    **Why is Revenue at {fmt_currency(revenue)}?**<br>
    The primary driver of revenue in this period is the **{top_prod}**, which accounts for 
    **{top_prod_pct:.1%}** of all sales. Performance is heavily skewed geographically, with 
    **{top_reg}** leading the market, while **{bot_reg}** is underperforming. 
    <br><br>
    **Risk Factors & Root Causes:**<br>
    Overall return rates sit at **{ret_rate:.1%}**. However, the **{high_ret_prod}** is experiencing unusually high 
    return rates compared to the baseline, which may indicate a quality control issue or a mismatch in customer expectations.
    """
    return story


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_currency(value: float, prefix: str = "₹") -> str:
    """Format a number as compact currency string (K / M / B)."""
    if value >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{prefix}{value / 1_000:.1f}K"
    else:
        return f"{prefix}{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{value:,.0f}"

# ---------------------------------------------------------------------------
# UI Theme Engine & HTML Components
# ---------------------------------------------------------------------------

def get_theme_css(theme: str = "dark") -> str:
    """Generate dynamic CSS based on the selected premium SaaS theme."""
    if theme == "light":
        css_vars = """
        --bg-gradient: radial-gradient(circle at 50% 0%, #FFFFFF, #F3F4F6);
        --sidebar-bg: linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
        --card-bg: rgba(255, 255, 255, 0.7);
        --card-border: rgba(99, 102, 241, 0.15);
        --text-primary: #111827;
        --text-secondary: #6B7280;
        --glow: rgba(99, 102, 241, 0.15);
        --border-gradient: linear-gradient(135deg, rgba(99,102,241,0.5), rgba(6,182,212,0.5));
        --positive: #22C55E;
        --negative: #F43F5E;
        --primary: #6366F1;
        """
    else:
        # Deep Navy / Indigo Premium Background
        css_vars = """
        --bg-gradient: radial-gradient(circle at 15% 0%, #1E1B4B 0%, #050816 60%, #000000 100%);
        --sidebar-bg: rgba(11, 16, 32, 0.85);
        --card-bg: rgba(17, 24, 39, 0.55);
        --card-border: rgba(255, 255, 255, 0.06);
        --text-primary: #F9FAFB;
        --text-secondary: #9CA3AF;
        --glow: rgba(99, 102, 241, 0.3);
        --border-gradient: linear-gradient(135deg, rgba(139,92,246,0.6), rgba(6,182,212,0.6));
        --positive: #10B981;
        --negative: #F43F5E;
        --primary: #818CF8;
        """

    return f"""
    <style>
    :root {{
        {css_vars}
    }}

    /* ── Global Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
        transition: background 0.5s ease, color 0.5s ease;
    }}

    /* ── App background ── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background: var(--bg-gradient) !important;
        background-attachment: fixed !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {{
        background: var(--sidebar-bg) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid var(--card-border);
    }}
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: var(--text-primary) !important;
    }}

    /* ── Streamlit Native Widget Overrides ── */
    label, [data-testid="stWidgetLabel"] p, .st-emotion-cache-1wivap2, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {{
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
    }}
    
    [data-baseweb="select"] > div, 
    [data-baseweb="input"] > div, 
    [data-baseweb="base-input"] {{
        background-color: var(--card-bg) !important;
        backdrop-filter: blur(12px);
        color: var(--text-primary) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }}
    [data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover {{
        border-color: rgba(99, 102, 241, 0.5) !important;
    }}

    /* Dropdown Menus */
    [data-baseweb="menu"], [data-baseweb="popover"] > div {{
        background-color: var(--sidebar-bg) !important;
        backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 8px;
    }}
    [data-baseweb="menu"] li {{
        color: var(--text-primary) !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background-color: rgba(99, 102, 241, 0.2) !important;
    }}

    /* Date Input */
    .stDateInput > div {{
        background-color: var(--card-bg) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }}

    /* Tags/Chips */
    [data-baseweb="tag"] {{
        background-color: rgba(99, 102, 241, 0.15) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
    }}
    [data-baseweb="tag"] span {{
        color: var(--text-primary) !important;
        font-weight: 500;
    }}

    /* ── Command Center Header ── */
    .command-header {{
        background: var(--card-bg);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 32px;
        box-shadow: 0 4px 24px -4px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .command-title {{
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
    }}
    .command-subtitle {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* ── Premium KPI Cards ── */
    .kpi-card {{
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        height: 100%;
        display: flex;
        flex-direction: column;
    }}
    .kpi-card::after {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 100%;
        background: radial-gradient(800px circle at var(--mouse-x, 50%) var(--mouse-y, 0%), rgba(255,255,255,0.04), transparent 40%);
        opacity: 0;
        transition: opacity 0.5s;
        pointer-events: none;
    }}
    .kpi-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -10px var(--glow), 0 1px 3px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1);
        border-color: rgba(139, 92, 246, 0.4);
    }}
    .kpi-card:hover::after {{
        opacity: 1;
    }}
    
    .kpi-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }}
    .kpi-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.03em;
    }}
    .kpi-icon-wrap {{
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--card-border);
        border-radius: 8px;
        padding: 6px 8px;
        font-size: 1.1rem;
    }}

    .kpi-hero-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }}
    /* Level 1 KPI gets special gradient text */
    .kpi-hero-value.level-1 {{
        background: linear-gradient(135deg, #FFFFFF 20%, var(--primary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
    }}

    .kpi-footer {{
        display: flex;
        align-items: center;
        margin-top: auto;
    }}
    .delta-badge {{
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }}
    .delta-badge.positive {{
        background: rgba(16, 185, 129, 0.15);
        color: var(--positive);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }}
    .delta-badge.negative {{
        background: rgba(244, 63, 94, 0.15);
        color: var(--negative);
        border: 1px solid rgba(244, 63, 94, 0.2);
    }}
    .delta-badge.neutral {{
        background: rgba(156, 163, 175, 0.15);
        color: var(--text-secondary);
        border: 1px solid rgba(156, 163, 175, 0.2);
    }}
    .kpi-subtext {{
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-left: 8px;
        font-weight: 500;
    }}

    /* ── Chart Containers ── */
    [data-testid="stPlotlyChart"] {{
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
        overflow: hidden;
    }}
    [data-testid="stPlotlyChart"]:hover {{
        border-color: rgba(139, 92, 246, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 10px 30px -10px var(--glow);
    }}

    /* ── Section headers ── */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--primary);
        margin: 40px 0 20px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .section-header::after {{
        content: "";
        flex-grow: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--card-border), transparent);
    }}
    </style>
    """

def render_kpi_card(title: str, value: str, icon: str = "") -> str:
    """Returns raw HTML for a highly animated KPI card."""
    return f"""
    <div class="kpi-card-custom">
        <div class="kpi-label">{icon} {title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """
