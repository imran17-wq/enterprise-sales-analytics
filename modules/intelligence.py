"""
intelligence.py
===============
Enterprise BI analytical backends for the Sales Analytics Dashboard.

Functions:
  - generate_executive_summary(df, kpis) -> dict
  - get_top_movers(df)                  -> dict
  - detect_anomalies(df)                -> dict
  - run_what_if(df, sliders)            -> dict
  - build_excel_export(df, kpis)        -> bytes
  - build_pdf_export(df, kpis, ...)     -> bytes
  - build_png_export(df)                -> bytes
"""

import io
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 — Executive Summary Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_executive_summary(df: pd.DataFrame, kpis: dict) -> dict:
    """
    Deterministically analyse the filtered DataFrame and return a structured
    executive summary with drivers, risks, and recommended actions.
    """
    if df.empty:
        return {"headline": "No data available.", "drivers": [], "risks": [], "actions": []}

    revenue   = kpis["total_revenue"]
    ret_rate  = kpis["return_rate"]

    # ── Revenue trend ────────────────────────────────────────────────────────
    monthly = df.groupby("year_month")["net_revenue"].sum().sort_index()
    if len(monthly) >= 2:
        mid       = max(1, len(monthly) // 2)
        first_h   = monthly.iloc[:mid].mean()
        second_h  = monthly.iloc[mid:].mean()
        trend_pct = ((second_h - first_h) / first_h * 100) if first_h > 0 else 0
        trend_dir = "increased" if trend_pct >= 0 else "declined"
        trend_str = f"Revenue {trend_dir} {abs(trend_pct):.1f}% in the recent period vs. the earlier period."
    else:
        trend_pct = 0
        trend_str = f"Total revenue stands at ₹{revenue:,.0f}."

    # ── Product analysis ─────────────────────────────────────────────────────
    prod_rev  = df.groupby("product")["net_revenue"].sum().sort_values(ascending=False)
    top_prod  = prod_rev.index[0]
    top_prod_pct = prod_rev.iloc[0] / revenue * 100 if revenue > 0 else 0

    # Product growth (first half vs second half by time)
    prod_drivers = []
    prod_risks   = []
    if len(monthly) >= 2:
        mid_ym = monthly.index[len(monthly) // 2]
        for prod in prod_rev.index[:5]:
            p_df = df[df["product"] == prod]
            p_m  = p_df.groupby("year_month")["net_revenue"].sum().sort_index()
            if len(p_m) >= 2:
                p_mid   = max(1, len(p_m) // 2)
                p_first = p_m.iloc[:p_mid].mean()
                p_sec   = p_m.iloc[p_mid:].mean()
                p_delta = ((p_sec - p_first) / p_first * 100) if p_first > 0 else 0
                if p_delta >= 5:
                    prod_drivers.append(f"{prod} revenue grew {p_delta:.1f}%")
                elif p_delta <= -5:
                    prod_risks.append(f"{prod} revenue declined {abs(p_delta):.1f}%")

    # ── Region analysis ──────────────────────────────────────────────────────
    reg_rev  = df.groupby("region")["net_revenue"].sum().sort_values(ascending=False)
    top_reg  = reg_rev.index[0]
    bot_reg  = reg_rev.index[-1]
    top_reg_pct = reg_rev.iloc[0] / revenue * 100 if revenue > 0 else 0

    # ── Promotion analysis ───────────────────────────────────────────────────
    promo_aov = df.groupby("promotion_used")["net_revenue"].mean()
    best_promo = promo_aov.idxmax() if not promo_aov.empty else "N/A"

    # ── Return rate analysis ─────────────────────────────────────────────────
    high_ret_prod = df.groupby("product")["returned"].mean()
    worst_ret_prod = high_ret_prod.idxmax() if not high_ret_prod.empty else "N/A"
    worst_ret_rate = high_ret_prod.max() * 100 if not high_ret_prod.empty else 0

    # ── Customer analysis ────────────────────────────────────────────────────
    cust_rev = df.groupby("customer_type")["net_revenue"].sum()
    top_cust = cust_rev.idxmax() if not cust_rev.empty else "N/A"

    # ── Build structured output ──────────────────────────────────────────────
    headline = f"{trend_str} {top_prod} leads product revenue with {top_prod_pct:.1f}% share. {top_reg} region is the top market."

    drivers = [
        f"{top_prod} contributes {top_prod_pct:.1f}% of total revenue",
        f"{top_reg} region generates {top_reg_pct:.1f}% of total revenue",
        f"{best_promo} is the highest-performing promotion by average order value",
        f"{top_cust} customers are the leading revenue segment",
    ] + prod_drivers[:2]

    risks = []
    if ret_rate > 10:
        risks.append(f"Return rate is elevated at {ret_rate:.1f}% — review quality for {worst_ret_prod} ({worst_ret_rate:.1f}% returns)")
    if trend_pct < -5:
        risks.append(f"Revenue trend is declining — immediate review recommended")
    risks.append(f"{bot_reg} region is underperforming vs. all other regions")
    risks += prod_risks[:2]

    actions = [
        f"Increase inventory allocation for {top_prod} to sustain momentum",
        f"Launch targeted campaigns in {bot_reg} to close the revenue gap",
        f"Scale the {best_promo} promotion — it demonstrates the highest customer AOV",
    ]
    if ret_rate > 8:
        actions.append(f"Initiate quality review for {worst_ret_prod} to reduce the {worst_ret_rate:.1f}% return rate")

    return {
        "headline":   headline,
        "drivers":    drivers,
        "risks":      risks,
        "actions":    actions,
        "trend_pct":  round(trend_pct, 1),
        "top_prod":   top_prod,
        "top_reg":    top_reg,
        "bot_reg":    bot_reg,
        "best_promo": best_promo,
        "return_rate": round(ret_rate, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — Top Movers Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def get_top_movers(df: pd.DataFrame) -> dict:
    """
    Compare first half vs second half of the time window to identify the
    top growing, declining, and key performers across products and regions.
    """
    if df.empty:
        return {}

    monthly = df.groupby("year_month")["net_revenue"].sum().sort_index()
    mid = max(1, len(monthly) // 2)

    # ── Product movers ───────────────────────────────────────────────────────
    prod_rows = []
    for prod in df["product"].unique():
        p_m = df[df["product"] == prod].groupby("year_month")["net_revenue"].sum().sort_index()
        if len(p_m) < 2:
            continue
        p_mid   = max(1, len(p_m) // 2)
        p_first = p_m.iloc[:p_mid].sum()
        p_sec   = p_m.iloc[p_mid:].sum()
        delta   = ((p_sec - p_first) / p_first * 100) if p_first > 0 else 0
        prod_rows.append({"product": prod, "delta_pct": delta,
                          "period1_rev": p_first, "period2_rev": p_sec})
    prod_mdf = pd.DataFrame(prod_rows).sort_values("delta_pct", ascending=False) if prod_rows else pd.DataFrame()

    # ── Region movers ────────────────────────────────────────────────────────
    reg_rows = []
    for reg in df["region"].unique():
        r_m = df[df["region"] == reg].groupby("year_month")["net_revenue"].sum().sort_index()
        if len(r_m) < 2:
            continue
        r_mid   = max(1, len(r_m) // 2)
        r_first = r_m.iloc[:r_mid].sum()
        r_sec   = r_m.iloc[r_mid:].sum()
        delta   = ((r_sec - r_first) / r_first * 100) if r_first > 0 else 0
        reg_rows.append({"region": reg, "delta_pct": delta,
                         "total_rev": df[df["region"] == reg]["net_revenue"].sum()})
    reg_mdf = pd.DataFrame(reg_rows).sort_values("delta_pct", ascending=False) if reg_rows else pd.DataFrame()

    # ── Return rate leaders ──────────────────────────────────────────────────
    ret_by_prod = df.groupby("product")["returned"].mean().sort_values(ascending=False)
    worst_ret   = ret_by_prod.index[0] if not ret_by_prod.empty else "N/A"

    # ── Best promotion ────────────────────────────────────────────────────────
    promo_aov = df.groupby("promotion_used")["net_revenue"].mean()
    best_promo = promo_aov.idxmax() if not promo_aov.empty else "N/A"
    best_promo_aov = promo_aov.max() if not promo_aov.empty else 0

    # ── Top revenue ───────────────────────────────────────────────────────────
    prod_total = df.groupby("product")["net_revenue"].sum()
    top_rev_prod = prod_total.idxmax() if not prod_total.empty else "N/A"

    reg_total = df.groupby("region")["net_revenue"].sum()
    top_rev_reg  = reg_total.idxmax() if not reg_total.empty else "N/A"
    bot_rev_reg  = reg_total.idxmin() if not reg_total.empty else "N/A"

    return {
        "top_growing_product":  prod_mdf.iloc[0]["product"]   if not prod_mdf.empty else "N/A",
        "top_growing_pct":      prod_mdf.iloc[0]["delta_pct"] if not prod_mdf.empty else 0,
        "top_declining_product":prod_mdf.iloc[-1]["product"]  if not prod_mdf.empty else "N/A",
        "top_declining_pct":    prod_mdf.iloc[-1]["delta_pct"]if not prod_mdf.empty else 0,
        "top_revenue_product":  top_rev_prod,
        "top_revenue_region":   top_rev_reg,
        "worst_revenue_region": bot_rev_reg,
        "highest_return_product": worst_ret,
        "highest_return_rate":  float(ret_by_prod.iloc[0] * 100) if not ret_by_prod.empty else 0,
        "best_promo":           best_promo,
        "best_promo_aov":       best_promo_aov,
        "product_movers":       prod_mdf,
        "region_movers":        reg_mdf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — What-If Simulator
# ─────────────────────────────────────────────────────────────────────────────

def run_what_if(df: pd.DataFrame, discount_delta: float,
                demand_delta: float, return_delta: float) -> dict:
    """
    Simple multiplicative scenario model.

    Parameters (all as percentage-point changes):
      discount_delta  : Δ discount %. Higher discount → lower net margin per unit.
      demand_delta    : Δ demand %. Higher demand → more orders.
      return_delta    : Δ return rate %. Higher returns → lower effective revenue.

    Returns projected KPIs.
    """
    if df.empty:
        return {}

    base_revenue = df["net_revenue"].sum()
    base_orders  = df["order_id"].nunique()
    base_aov     = df["net_revenue"].mean()
    base_returns = df["returned"].mean() * 100

    # Demand multiplier: +1% demand → +1% revenue
    demand_mult = 1 + demand_delta / 100

    # Discount penalty: every +1% discount cuts margin by ~1.5% (elasticity approx)
    discount_mult = 1 - (discount_delta / 100) * 1.5

    # Return penalty: +1% return rate → −0.8% effective revenue (partial refund model)
    return_mult = 1 - (return_delta / 100) * 0.8

    proj_revenue = base_revenue * demand_mult * max(discount_mult, 0.5) * max(return_mult, 0.5)
    proj_orders  = base_orders  * demand_mult
    proj_aov     = proj_revenue / proj_orders if proj_orders > 0 else 0
    proj_returns = max(0, base_returns + return_delta)

    delta_rev = proj_revenue - base_revenue
    delta_pct = delta_rev / base_revenue * 100 if base_revenue > 0 else 0

    return {
        "base_revenue":   base_revenue,
        "proj_revenue":   proj_revenue,
        "delta_revenue":  delta_rev,
        "delta_pct":      delta_pct,
        "base_orders":    base_orders,
        "proj_orders":    proj_orders,
        "base_aov":       base_aov,
        "proj_aov":       proj_aov,
        "base_returns":   base_returns,
        "proj_returns":   proj_returns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6 — Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Run anomaly detection on monthly aggregates using:
      1. Z-score (|z| > 2) on revenue
      2. IsolationForest on [revenue, orders, return_rate]

    Returns a dict with:
      - monthly_df: DataFrame with anomaly flags & z-scores
      - anomalies:  list of structured anomaly dicts for display
    """
    if df.empty or "year_month" not in df.columns:
        return {"monthly_df": pd.DataFrame(), "anomalies": []}

    monthly = (
        df.groupby("year_month")
        .agg(
            revenue=("net_revenue",  "sum"),
            orders=("order_id",      "nunique"),
            return_rate=("returned", "mean"),
        )
        .reset_index()
        .sort_values("year_month")
    )

    if len(monthly) < 4:
        return {"monthly_df": monthly, "anomalies": []}

    # ── Z-score ──────────────────────────────────────────────────────────────
    monthly["rev_z"]    = scipy_stats.zscore(monthly["revenue"].fillna(0))
    monthly["order_z"]  = scipy_stats.zscore(monthly["orders"].fillna(0))
    monthly["return_z"] = scipy_stats.zscore(monthly["return_rate"].fillna(0))

    # ── IsolationForest ───────────────────────────────────────────────────────
    feat_cols = ["revenue", "orders", "return_rate"]
    X = monthly[feat_cols].fillna(0).values
    iso = IsolationForest(contamination=0.15, random_state=42)
    monthly["iso_flag"] = iso.fit_predict(X)  # -1 = anomaly, 1 = normal

    # ── Combine: flag row if z-score OR isolation forest fires ───────────────
    monthly["is_anomaly"] = (
        (monthly["rev_z"].abs() > 2) |
        (monthly["order_z"].abs() > 2) |
        (monthly["return_z"].abs() > 2) |
        (monthly["iso_flag"] == -1)
    )

    # ── Build structured anomaly list ─────────────────────────────────────────
    anomalies = []
    for _, row in monthly[monthly["is_anomaly"]].iterrows():
        reasons = []
        if abs(row["rev_z"]) > 2:
            direction = "spike" if row["rev_z"] > 0 else "drop"
            reasons.append(f"Revenue {direction} (z={row['rev_z']:.1f})")
        if abs(row["order_z"]) > 2:
            direction = "surge" if row["order_z"] > 0 else "drop"
            reasons.append(f"Order count {direction} (z={row['order_z']:.1f})")
        if abs(row["return_z"]) > 2:
            direction = "spike" if row["return_z"] > 0 else "drop"
            reasons.append(f"Return rate {direction} (z={row['return_z']:.1f})")
        if row["iso_flag"] == -1 and not reasons:
            reasons.append("Statistical outlier detected by Isolation Forest")

        severity = "🔴 High" if abs(row["rev_z"]) > 2.5 else "🟡 Medium"
        anomalies.append({
            "month":    row["year_month"],
            "revenue":  row["revenue"],
            "orders":   row["orders"],
            "ret_rate": row["return_rate"] * 100,
            "reasons":  reasons,
            "severity": severity,
            "rev_z":    row["rev_z"],
        })

    return {"monthly_df": monthly, "anomalies": anomalies}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7 — Excel Export Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_excel_export(df: pd.DataFrame, kpis: dict, filters: dict) -> bytes:
    """
    Build a multi-sheet Excel workbook and return it as bytes.

    Sheets:
      1. KPI Summary
      2. Product Summary
      3. Region Summary
      4. Monthly Trends  [NEW]
      5. Promotions Analysis  [NEW]
      6. Raw Data (up to 5,000 rows)
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── Formats ──────────────────────────────────────────────────────────
        hdr_fmt  = wb.add_format({"bold": True, "bg_color": "#6C63FF",
                                  "font_color": "#FFFFFF", "border": 1,
                                  "align": "center", "valign": "vcenter"})
        num_fmt  = wb.add_format({"num_format": "#,##0.00", "border": 1})
        int_fmt  = wb.add_format({"num_format": "#,##0",    "border": 1})
        pct_fmt  = wb.add_format({"num_format": "0.00%",    "border": 1})
        txt_fmt  = wb.add_format({"border": 1})
        date_fmt = wb.add_format({"num_format": "yyyy-mm-dd", "border": 1})
        title_fmt = wb.add_format({"bold": True, "font_size": 14,
                                   "font_color": "#6C63FF"})
        meta_fmt = wb.add_format({"italic": True, "font_color": "#7F8C8D",
                                  "font_size": 9})

        def auto_col_width(ws, df_in, start_row=2):
            for i, col in enumerate(df_in.columns):
                max_len = max(
                    len(str(col)),
                    df_in[col].astype(str).str.len().max() if len(df_in) > 0 else 0
                )
                ws.set_column(i, i, min(max(max_len + 2, 12), 40))

        def write_sheet(sheet_name, data_df, title=""):
            ws = writer.sheets.get(sheet_name)
            data_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            ws = writer.sheets[sheet_name]
            if title:
                ws.write(0, 0, title, title_fmt)
            # Build a compact filter summary string
            filter_parts = [f"{k.replace('_',' ').title()}: {len(v)} selected" for k, v in filters.items() if v]
            filter_str = "  |  ".join(filter_parts) if filter_parts else "No filters applied"
            ws.write(1, 0, f"Generated from {len(df):,} filtered rows  ·  {filter_str}", meta_fmt)
            for col_num, col_name in enumerate(data_df.columns):
                ws.write(2, col_num, col_name, hdr_fmt)
            auto_col_width(ws, data_df)

        # ── Sheet 1: KPI Summary ─────────────────────────────────────────────
        kpi_df = pd.DataFrame([
            {"Metric": "Total Revenue (₹)",   "Value": round(kpis.get("total_revenue", 0), 2),   "Format": "currency"},
            {"Metric": "Total Profit (₹)",    "Value": round(kpis.get("total_profit", 0), 2),    "Format": "currency"},
            {"Metric": "Total Orders",         "Value": int(kpis.get("total_orders", 0)),          "Format": "integer"},
            {"Metric": "Total Qty Sold",       "Value": int(kpis.get("total_quantity", 0)),        "Format": "integer"},
            {"Metric": "Avg Order Value (₹)",  "Value": round(kpis.get("avg_order_value", 0), 2), "Format": "currency"},
            {"Metric": "Return Rate (%)",      "Value": round(kpis.get("return_rate", 0), 2),     "Format": "percent"},
            {"Metric": "Health Score (/100)",  "Value": round(kpis.get("health_score", 0), 1),    "Format": "number"},
            {"Metric": "Growth Score (/100)",  "Value": round(kpis.get("growth_score", 0), 1),    "Format": "number"},
            {"Metric": "Best Region",          "Value": kpis.get("best_region", ""),              "Format": "text"},
            {"Metric": "Best Product",         "Value": kpis.get("best_product", ""),             "Format": "text"},
        ])
        # Write KPI sheet manually with per-row number formatting
        kpi_df[["Metric", "Value"]].to_excel(writer, sheet_name="KPI Summary", index=False, startrow=2)
        ws_kpi = writer.sheets["KPI Summary"]
        ws_kpi.write(0, 0, "📊 KPI Summary", title_fmt)
        filter_parts_kpi = [f"{k.replace('_',' ').title()}: {len(v)} selected" for k, v in filters.items() if v]
        ws_kpi.write(1, 0, "Generated from {:,} filtered rows  ·  {}".format(
            len(df), "  |  ".join(filter_parts_kpi) if filter_parts_kpi else "No filters"), meta_fmt)
        ws_kpi.write(2, 0, "Metric", hdr_fmt)
        ws_kpi.write(2, 1, "Value",  hdr_fmt)
        fmt_map = {"currency": num_fmt, "integer": int_fmt, "percent": num_fmt, "number": num_fmt, "text": txt_fmt}
        for row_i, (_, row) in enumerate(kpi_df.iterrows(), start=3):
            ws_kpi.write(row_i, 0, row["Metric"], txt_fmt)
            ws_kpi.write(row_i, 1, row["Value"],  fmt_map.get(row["Format"], txt_fmt))
        ws_kpi.set_column(0, 0, 28)
        ws_kpi.set_column(1, 1, 22)

        # ── Sheet 2: Product Summary ──────────────────────────────────────────
        prod_df = (
            df.groupby("product")
            .agg(
                Total_Revenue=("net_revenue",  "sum"),
                Total_Orders=("order_id",      "nunique"),
                Total_Qty_Sold=("quantity_sold","sum"),
                Avg_Order_Value=("net_revenue", "mean"),
                Avg_Unit_Price=("unit_price",   "mean"),
                Avg_Discount=("discount",       "mean"),
                Return_Rate=("returned",        "mean"),
            )
            .reset_index()
            .sort_values("Total_Revenue", ascending=False)
        )
        prod_df["Return_Rate"]  = (prod_df["Return_Rate"] * 100).round(2)
        prod_df["Avg_Discount"] = (prod_df["Avg_Discount"] * 100).round(2)
        prod_df.columns         = [c.replace("_", " ") for c in prod_df.columns]
        write_sheet("Product Summary", prod_df, "📦 Product Performance")

        # ── Sheet 3: Region Summary ───────────────────────────────────────────
        reg_df = (
            df.groupby("region")
            .agg(
                Total_Revenue=("net_revenue",  "sum"),
                Total_Orders=("order_id",      "nunique"),
                Total_Qty_Sold=("quantity_sold","sum"),
                Avg_Order_Value=("net_revenue", "mean"),
                Return_Rate=("returned",        "mean"),
            )
            .reset_index()
            .sort_values("Total_Revenue", ascending=False)
        )
        reg_df["Return_Rate"] = (reg_df["Return_Rate"] * 100).round(2)
        reg_df.columns        = [c.replace("_", " ") for c in reg_df.columns]
        write_sheet("Region Summary", reg_df, "🗺️ Region Performance")

        # ── Sheet 4: Monthly Trends ───────────────────────────────────────────
        monthly_df = (
            df.groupby("year_month")
            .agg(
                Total_Revenue=("net_revenue",   "sum"),
                Total_Orders=("order_id",       "nunique"),
                Total_Qty_Sold=("quantity_sold", "sum"),
                Avg_Order_Value=("net_revenue",  "mean"),
                Return_Rate=("returned",         "mean"),
                Avg_Discount=("discount",        "mean"),
            )
            .reset_index()
            .sort_values("year_month")
        )
        monthly_df["Return_Rate"] = (monthly_df["Return_Rate"] * 100).round(2)
        monthly_df["Avg_Discount"] = (monthly_df["Avg_Discount"] * 100).round(2)
        monthly_df["MoM_Revenue_Change_%"] = monthly_df["Total_Revenue"].pct_change().mul(100).round(2)
        monthly_df.columns = [c.replace("_", " ") for c in monthly_df.columns]
        write_sheet("Monthly Trends", monthly_df, "📅 Monthly Sales Trends")

        # ── Sheet 5: Promotions Analysis ──────────────────────────────────────
        promo_df = (
            df.groupby("promotion_used")
            .agg(
                Total_Revenue=("net_revenue",   "sum"),
                Total_Orders=("order_id",       "nunique"),
                Avg_Order_Value=("net_revenue",  "mean"),
                Total_Qty_Sold=("quantity_sold", "sum"),
                Return_Rate=("returned",         "mean"),
                Avg_Discount=("discount",        "mean"),
            )
            .reset_index()
            .sort_values("Total_Revenue", ascending=False)
        )
        promo_df["Return_Rate"] = (promo_df["Return_Rate"] * 100).round(2)
        promo_df["Avg_Discount"] = (promo_df["Avg_Discount"] * 100).round(2)
        promo_df["Revenue_Share_%"] = (promo_df["Total_Revenue"] / promo_df["Total_Revenue"].sum() * 100).round(2)
        promo_df.columns = [c.replace("_", " ") for c in promo_df.columns]
        write_sheet("Promotions Analysis", promo_df, "🎁 Promotions Analysis")

        # ── Sheet 6: Raw Data ─────────────────────────────────────────────────
        raw_cols = ["date", "order_id", "region", "product", "salesperson",
                    "customer_name", "customer_type", "quantity_sold", "unit_price",
                    "discount", "net_revenue", "shipping_cost", "payment_method",
                    "promotion_used", "returned", "store_location", "region_manager"]
        raw_cols = [c for c in raw_cols if c in df.columns]
        raw_df   = df[raw_cols].head(5000).reset_index(drop=True)
        write_sheet("Raw Data", raw_df, f"🔎 Raw Data (up to 5,000 rows)")

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8 — PDF Report Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf_export(df: pd.DataFrame, kpis: dict, filters: dict,
                     summary: dict, date_str: str = "") -> bytes:
    """
    Generate a formatted PDF sales report using fpdf2.
    Includes: KPI summary, product table, region table, executive summary.
    Returns bytes suitable for st.download_button.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required for PDF export. Add 'fpdf2>=2.7.0' to requirements.txt")

    import datetime

    BRAND   = (108, 99, 255)   # primary indigo
    LIGHT   = (245, 245, 250)
    DARK    = (30, 30, 45)
    WHITE   = (255, 255, 255)
    GRAY    = (120, 120, 140)
    GREEN   = (46, 204, 113)
    RED     = (231, 76, 60)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    def fmt_inr(val):
        try:
            return f"Rs {float(val):,.0f}"
        except Exception:
            return str(val)

    def section(title: str):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*BRAND)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*BRAND)
        pdf.set_line_width(0.5)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(4)
        pdf.set_text_color(30, 30, 45)

    def table(headers, rows, col_widths=None):
        if col_widths is None:
            avail = 174
            col_widths = [avail // len(headers)] * len(headers)
        # Header row
        pdf.set_fill_color(*BRAND)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 9)
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 8, str(h), border=1, fill=True, align="C")
        pdf.ln()
        # Data rows
        pdf.set_font("Helvetica", "", 8)
        for r_idx, row in enumerate(rows):
            pdf.set_fill_color(*LIGHT) if r_idx % 2 == 0 else pdf.set_fill_color(*WHITE)
            pdf.set_text_color(*DARK)
            for i, cell in enumerate(row):
                pdf.cell(col_widths[i], 7, str(cell)[:35], border=1, fill=True, align="C")
            pdf.ln()
        pdf.ln(4)

    # ── Cover Page ─────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*BRAND)
    pdf.rect(0, 0, 210, 60, "F")
    pdf.set_y(18)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 12, "Sales Intelligence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, "Enterprise Analytics Dashboard", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    if date_str:
        pdf.cell(0, 6, f"Date Range: {date_str}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(70)
    pdf.set_text_color(*DARK)

    # ── KPI Summary ────────────────────────────────────────────────────────────
    section("Key Performance Indicators")
    kpi_rows = [
        ("Total Revenue",    fmt_inr(kpis.get("total_revenue", 0))),
        ("Total Profit",     fmt_inr(kpis.get("total_profit", 0))),
        ("Total Orders",     f"{int(kpis.get('total_orders', 0)):,}"),
        ("Total Qty Sold",   f"{int(kpis.get('total_quantity', 0)):,}"),
        ("Avg Order Value",  fmt_inr(kpis.get("avg_order_value", 0))),
        ("Return Rate",      f"{kpis.get('return_rate', 0):.2f}%"),
        ("Best Region",      str(kpis.get("best_region", "N/A"))),
        ("Best Product",     str(kpis.get("best_product", "N/A"))),
        ("Health Score",     f"{kpis.get('health_score', 0):.0f}/100"),
        ("Growth Score",     f"{kpis.get('growth_score', 0):.0f}/100"),
    ]
    table(["Metric", "Value"], kpi_rows, [90, 84])

    # ── Filter Summary ─────────────────────────────────────────────────────────
    section("Applied Filters")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY)
    for k, v in filters.items():
        label = k.replace("_", " ").title()
        vals  = ", ".join(map(str, v)) if v else "None"
        pdf.cell(0, 6, f"{label}: {vals}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*DARK)
    pdf.cell(0, 6, f"Filtered Records: {len(df):,}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Product Summary ────────────────────────────────────────────────────────
    pdf.add_page()
    section("Product Performance")
    prod_df = (
        df.groupby("product")
        .agg(Revenue=("net_revenue", "sum"), Orders=("order_id", "nunique"),
             ReturnRate=("returned", "mean"))
        .reset_index().sort_values("Revenue", ascending=False)
    )
    prod_rows = [
        (r["product"], fmt_inr(r["Revenue"]), f"{int(r['Orders']):,}", f"{r['ReturnRate']*100:.1f}%")
        for _, r in prod_df.iterrows()
    ]
    table(["Product", "Revenue", "Orders", "Return Rate"], prod_rows, [50, 50, 37, 37])

    # ── Region Summary ─────────────────────────────────────────────────────────
    section("Region Performance")
    reg_df = (
        df.groupby("region")
        .agg(Revenue=("net_revenue", "sum"), Orders=("order_id", "nunique"),
             ReturnRate=("returned", "mean"))
        .reset_index().sort_values("Revenue", ascending=False)
    )
    reg_rows = [
        (r["region"], fmt_inr(r["Revenue"]), f"{int(r['Orders']):,}", f"{r['ReturnRate']*100:.1f}%")
        for _, r in reg_df.iterrows()
    ]
    table(["Region", "Revenue", "Orders", "Return Rate"], reg_rows, [50, 50, 37, 37])

    # ── Executive Summary ──────────────────────────────────────────────────────
    pdf.add_page()
    section("Executive Intelligence Summary")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 6, summary.get("headline", ""), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    for heading, key in [("Key Drivers", "drivers"), ("Risks", "risks"), ("Recommended Actions", "actions")]:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*BRAND)
        pdf.cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*DARK)
        for item in summary.get(key, []):
            bullet = "-> " if heading == "Recommended Actions" else "* "
            pdf.multi_cell(0, 5.5, f"  {bullet}{item}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# Feature 9 — PNG Chart Export
# ─────────────────────────────────────────────────────────────────────────────

def build_png_export(df: pd.DataFrame) -> bytes:
    """
    Render a composite 2x2 Plotly chart image (revenue trend, region pie,
    product bar, return rate bar) and return it as PNG bytes.
    Requires kaleido>=0.2.1.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required for PNG export.")

    monthly = df.groupby("year_month")["net_revenue"].sum().reset_index()
    region  = df.groupby("region")["net_revenue"].sum().reset_index()
    product = df.groupby("product")["net_revenue"].sum().reset_index().sort_values("net_revenue", ascending=False)
    ret     = df.groupby("product")["returned"].mean().reset_index()
    ret["returned"] = ret["returned"] * 100

    COLORS = ["#6C63FF", "#43E8D8", "#FF6584", "#F39C12", "#2ECC71", "#D946EF", "#818CF8"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Monthly Revenue Trend", "Revenue by Region",
                        "Revenue by Product",    "Return Rate by Product"],
        specs=[[{"type": "scatter"}, {"type": "pie"}],
               [{"type": "bar"},    {"type": "bar"}]],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # Monthly trend
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["net_revenue"],
        mode="lines+markers", line=dict(color="#6C63FF", width=2.5),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.12)", name="Revenue"
    ), row=1, col=1)

    # Region pie
    fig.add_trace(go.Pie(
        labels=region["region"], values=region["net_revenue"],
        hole=0.45, marker=dict(colors=COLORS),
        textinfo="label+percent", showlegend=False
    ), row=1, col=2)

    # Product bar
    fig.add_trace(go.Bar(
        x=product["product"], y=product["net_revenue"],
        marker_color=COLORS[:len(product)], showlegend=False
    ), row=2, col=1)

    # Return rate bar
    fig.add_trace(go.Bar(
        x=ret["product"], y=ret["returned"],
        marker_color="#FF6584", showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        title=dict(text="Sales Intelligence Dashboard — Export Snapshot",
                   font=dict(size=16, color="#6C63FF")),
        paper_bgcolor="#1A1D27",
        plot_bgcolor="#1A1D27",
        font=dict(family="Arial, sans-serif", color="#EAEAEA", size=11),
        height=900, width=1400,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", color="#EAEAEA")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", color="#EAEAEA")

    try:
        return fig.to_image(format="png", scale=2)
    except Exception as e:
        raise RuntimeError(
            f"PNG export failed. Ensure kaleido>=0.2.1 is installed. Error: {e}"
        )
