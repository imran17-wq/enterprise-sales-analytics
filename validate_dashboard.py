"""validate_dashboard.py - Tests all dashboard components without starting Streamlit."""
import sys
import io
import warnings
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from data_loader import load_raw_data
from data_cleaning import clean_data
from feature_engineering import (
    engineer_features, build_monthly_sales, build_region_summary,
    build_product_summary, build_pivot_region_product, build_customer_summary,
    build_payment_summary, build_promotion_summary, build_return_heatmap,
    build_discount_revenue,
)
from forecasting import train_forecast_model, forecast_future, analyze_product_growth
from utils import (
    compute_kpis, generate_insights, fmt_currency,
    CHART_LAYOUT, PRODUCT_COLORS, REGION_COLORS,
)
import plotly.express as px
import plotly.graph_objects as go

print("Loading and cleaning data...")
raw   = load_raw_data()
clean = clean_data(raw)
df    = engineer_features(clean)
print(f"  Shape: {df.shape}  OK")

print("\nBuilding aggregation tables...")
monthly  = build_monthly_sales(df)
region   = build_region_summary(df)
product  = build_product_summary(df)
pivot    = build_pivot_region_product(df)
customer = build_customer_summary(df)
payment  = build_payment_summary(df)
promo    = build_promotion_summary(df)
heatmap  = build_return_heatmap(df)
scatter  = build_discount_revenue(df)
print("  All 9 tables: OK")

print("\nBuilding all 11 Plotly charts...")

# Chart 1 - Monthly trend
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["total_revenue"], mode="lines"))
print("  Chart 1 (Monthly Trend): OK")

# Chart 2 - Region donut
fig2 = go.Figure(go.Pie(labels=region["region"], values=region["total_revenue"], hole=0.55))
print("  Chart 2 (Region Donut): OK")

# Chart 3 - Product performance
fig3 = go.Figure()
fig3.add_trace(go.Bar(x=product["product"], y=product["total_revenue"]))
fig3.add_trace(go.Scatter(x=product["product"], y=product["return_rate"]*100, mode="lines+markers", yaxis="y2"))
print("  Chart 3 (Product Performance): OK")

# Chart 4 - Customer
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=customer["customer_type"], y=customer["total_revenue"]))
print("  Chart 4 (Customer Distribution): OK")

# Chart 5 - Payment (Plotly 6 compatible - use list for hover_data)
fig5 = px.bar(payment, x="payment_method", y="total_revenue", color="payment_method",
              text_auto=".2s", color_discrete_sequence=PRODUCT_COLORS)
print("  Chart 5 (Payment Method): OK")

# Chart 6 - Promotion (FIXED: use list not dict with format strings for hover_data)
fig6 = px.bar(
    promo, x="description", y="total_revenue", color="avg_discount",
    text_auto=".2s",
    color_continuous_scale=[[0, "#6C63FF"], [0.5, "#43E8D8"], [1, "#FF6584"]],
    hover_data=["return_rate", "avg_quantity"],
)
print("  Chart 6 (Promotion): OK")

# Chart 7 - Scatter
fig7 = px.scatter(
    scatter, x="discount_pct", y="net_revenue", color="product",
    size="quantity_sold", symbol="customer_type", opacity=0.75,
    color_discrete_sequence=PRODUCT_COLORS,
    hover_data=["region", "customer_type", "quantity_sold"],
)
print("  Chart 7 (Scatter): OK")

# Chart 8 - Heatmap
fig8 = go.Figure(go.Heatmap(
    z=heatmap.values, x=heatmap.columns.tolist(), y=heatmap.index.tolist()
))
print("  Chart 8 (Return Heatmap): OK")

# Chart 9 - Top products
top10 = df.groupby("product")["net_revenue"].sum().reset_index().sort_values("net_revenue")
fig9 = go.Figure(go.Bar(x=top10["net_revenue"], y=top10["product"], orientation="h"))
print("  Chart 9 (Top Products): OK")

# Chart 10 - Histogram
fig10 = go.Figure()
for prod, color in zip(df["product"].unique(), PRODUCT_COLORS):
    fig10.add_trace(go.Histogram(
        x=df[df["product"] == prod]["net_revenue"], name=prod, opacity=0.7, marker_color=color
    ))
print("  Chart 10 (Histogram): OK")

# Chart 11 - Correlation
corr_cols = ["unit_price", "quantity_sold", "discount", "total_sales",
             "net_revenue", "discount_amount", "shipping_cost", "returned"]
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr().round(3)
fig11 = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist()))
print("  Chart 11 (Correlation Matrix): OK")

print("\nComputing KPIs and insights...")
kpis     = compute_kpis(df)
insights = generate_insights(df, kpis)
print(f"  KPIs: {len(kpis)} keys  |  Insights: {len(insights)}")

print("\nRunning ML forecasting...")
result = train_forecast_model(df)
future = forecast_future(result, n_months=6)
print(f"  Metrics: {result['metrics']}")
print(f"  Forecast rows: {len(future)}")

print("\nRunning Product ML Growth analysis...")
growth_df = analyze_product_growth(df, n_months=6)
print(f"  Hot Products identified: {len(growth_df)}")
if not growth_df.empty:
    print(f"  Top Hot Take: {growth_df.iloc[0]['product']} (+{growth_df.iloc[0]['growth_pct']:.1f}%)")

print("\n" + "="*50)
print("ALL DASHBOARD COMPONENTS VALIDATED SUCCESSFULLY!")
print("="*50)
