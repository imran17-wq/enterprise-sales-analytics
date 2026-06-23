import pandas as pd
import numpy as np
# ---------------------------------------------------------------------------
# Revenue category thresholds (percentile-based, set at runtime)
# ---------------------------------------------------------------------------
REVENUE_BINS   = [0, 50_000, 150_000, 350_000, np.inf]
REVENUE_LABELS = ["Low", "Medium", "High", "Premium"]


# ---------------------------------------------------------------------------
# Derived column creators
# ---------------------------------------------------------------------------

def add_revenue_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["revenue"]         = df["unit_price"] * df["quantity_sold"]
    df["discount_amount"] = df["revenue"] * df["discount"]
    df["net_revenue"]     = df["revenue"] - df["discount_amount"]
    return df


def add_return_status(df: pd.DataFrame) -> pd.DataFrame:
    df["return_status"] = df["returned"].map({0: "Not Returned", 1: "Returned"})
    return df
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month"]      = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["quarter"]    = df["date"].dt.quarter.map(
        {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    )
    df["year"]       = df["date"].dt.year
    df["month_year"] = df["date"].dt.to_period("M").astype(str)
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    return df
def add_revenue_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bin net_revenue into Revenue Category labels."""
    df["revenue_category"] = pd.cut(
        df["net_revenue"],
        bins=REVENUE_BINS,
        labels=REVENUE_LABELS,
        right=True,
    )
    return df
def add_delivery_days(df: pd.DataFrame) -> pd.DataFrame:
    if "delivery_date" in df.columns and "order_date" in df.columns:
        df["delivery_days"] = (
            df["delivery_date"] - df["order_date"]
        ).dt.days.clip(lower=0)
    return df
# ---------------------------------------------------------------------------
# Aggregated tables (using groupby / pivot_table / merge / apply / transform)
# ---------------------------------------------------------------------------
def build_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby("year_month", sort=True)
        .agg(
            total_revenue  = ("net_revenue",  "sum"),
            total_orders   = ("order_id",     "count"),
            total_quantity = ("quantity_sold", "sum"),
            avg_order_value= ("net_revenue",  "mean"),
        )
        .reset_index()
    )
    # Add month-over-month growth using apply()
    monthly["mom_growth"] = monthly["total_revenue"].pct_change().apply(
        lambda x: round(x * 100, 2) if pd.notnull(x) else 0.0
    )
    return monthly


def build_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    region = (
        df.groupby("region")
        .agg(
            total_revenue  = ("net_revenue",  "sum"),
            total_orders   = ("order_id",     "count"),
            total_quantity = ("quantity_sold", "sum"),
            avg_discount   = ("discount",      "mean"),
            return_rate    = ("returned",      "mean"),
        )
        .reset_index()
    )
    # transform()-style share column
    region["revenue_share_pct"] = (
        region["total_revenue"] / region["total_revenue"].sum() * 100
    ).round(2)
    region = region.sort_values("total_revenue", ascending=False)
    return region


def build_product_summary(df: pd.DataFrame) -> pd.DataFrame:
    product = (
        df.groupby("product")
        .agg(
            total_revenue  = ("net_revenue",  "sum"),
            total_orders   = ("order_id",     "count"),
            total_quantity = ("quantity_sold", "sum"),
            avg_unit_price = ("unit_price",   "mean"),
            avg_discount   = ("discount",      "mean"),
            return_rate    = ("returned",      "mean"),
        )
        .reset_index()
    )
    # apply() to compute revenue per unit
    product["revenue_per_unit"] = product.apply(
        lambda r: r["total_revenue"] / r["total_quantity"]
        if r["total_quantity"] > 0 else 0,
        axis=1,
    )
    product = product.sort_values("total_revenue", ascending=False)
    return product
def build_pivot_region_product(df: pd.DataFrame) -> pd.DataFrame:
    """
    pivot_table() : Revenue by Region × Product matrix.
    """
    pivot = pd.pivot_table(
        df,
        values="net_revenue",
        index="region",
        columns="product",
        aggfunc="sum",
        fill_value=0,
    ).round(2)
    return pivot


def build_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Customer type analysis using groupby()."""
    cust = (
        df.groupby("customer_type")
        .agg(
            total_revenue  = ("net_revenue",  "sum"),
            total_orders   = ("order_id",     "count"),
            avg_order_value= ("net_revenue",  "mean"),
            return_rate    = ("returned",      "mean"),
            avg_discount   = ("discount",      "mean"),
        )
        .reset_index()
    )
    return cust
def build_payment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Payment method breakdown using groupby()."""
    pay = (
        df.groupby("payment_method")
        .agg(
            total_revenue = ("net_revenue", "sum"),
            order_count   = ("order_id",    "count"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    return pay


def build_promotion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Promotion effectiveness using groupby() + merge()."""
    promo = (
        df.groupby("promotion_used")
        .agg(
            total_revenue  = ("net_revenue",   "sum"),
            order_count    = ("order_id",      "count"),
            avg_discount   = ("discount",       "mean"),
            avg_quantity   = ("quantity_sold",  "mean"),
            return_rate    = ("returned",       "mean"),
        )
        .reset_index()
    )
    # merge() to join with an enriched promo description table
    promo_info = pd.DataFrame({
        "promotion_used": ["NONE", "FREESHIP", "SAVE10", "WINTER15", "SUMMER20"],
        "description":    [
            "No Promotion",
            "Free Shipping",
            "10% Savings",
            "15% Winter Deal",
            "20% Summer Deal",
        ],
    })
    promo = promo.merge(promo_info, on="promotion_used", how="left")
    promo["description"].fillna(promo["promotion_used"], inplace=True)
    return promo.sort_values("total_revenue", ascending=False)


def build_return_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rate by Region × Product using pivot_table()
    (used for the heatmap visualisation).
    """
    heatmap = pd.pivot_table(
        df,
        values="returned",
        index="region",
        columns="product",
        aggfunc="mean",
        fill_value=0,
    ).round(4)
    return heatmap


def build_discount_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Discount vs Revenue scatter data — order-level."""
    scatter = df[["discount", "net_revenue", "product", "region",
                  "customer_type", "quantity_sold"]].copy()
    scatter["discount_pct"] = (scatter["discount"] * 100).round(0).astype(int)
    return scatter


def add_region_avg_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrates transform(): add each order's region average revenue
    as a new column 'region_avg_revenue'.
    """
    df["region_avg_revenue"] = df.groupby("region")["net_revenue"].transform("mean")
    return df


# ---------------------------------------------------------------------------
# Master feature pipeline
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame  – cleaned data from data_cleaning

    Returns
    -------
    pd.DataFrame  – enriched data ready for dashboard + ML
    """
    df = add_revenue_columns(df)
    df = add_return_status(df)
    df = add_time_features(df)
    df = add_revenue_category(df)
    df = add_delivery_days(df)
    df = add_region_avg_revenue(df)
    print(f"[feature_engineering] Done. Feature-engineered shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader  import load_raw_data
    from data_cleaning import clean_data

    raw     = load_raw_data()
    cleaned = clean_data(raw)
    feat    = engineer_features(cleaned)

    print("\n--- Monthly Sales ---")
    print(build_monthly_sales(feat).head())

    print("\n--- Region Summary ---")
    print(build_region_summary(feat))

    print("\n--- Pivot: Region x Product ---")
    print(build_pivot_region_product(feat))
