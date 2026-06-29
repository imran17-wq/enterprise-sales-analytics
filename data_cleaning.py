import pandas as pd
import numpy as np
# ---------------------------------------------------------------------------
# Column name mapping  (raw → standardised)
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    "Date":           "date",
    "Region":         "region",
    "Product":        "product",
    "Quantity":       "quantity_sold",
    "UnitPrice":      "unit_price",
    "StoreLocation":  "store_location",
    "CustomerType":   "customer_type",
    "Discount":       "discount",
    "Salesperson":    "salesperson",
    "TotalPrice":     "total_sales",
    "PaymentMethod":  "payment_method",
    "Promotion":      "promotion_used",
    "Returned":       "returned",
    "OrderID":        "order_id",
    "CustomerName":   "customer_name",
    "ShippingCost":   "shipping_cost",
    "OrderDate":      "order_date",
    "DeliveryDate":   "delivery_date",
    "RegionManager":  "region_manager",
}
# Numeric columns where outliers will be treated
NUMERIC_OUTLIER_COLS = ["unit_price", "total_sales", "quantity_sold", "shipping_cost"]
# Text columns and their expected title-case or known categories
TEXT_STANDARDISE = {
    "region":        str.title,
    "product":       str.title,
    "customer_type": str.title,
    "payment_method": str.title,
    "salesperson":   str.title,
}
DATE_COLS = ["date", "order_date", "delivery_date"]
# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to snake_case standardised names."""
    df = df.rename(columns=COLUMN_RENAME_MAP)
    # Also lowercase any remaining column names not in the map
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[cleaning] Standardised columns: {list(df.columns)}")
    return df
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols    = df.select_dtypes(include=["object"]).columns.tolist()
    date_cols    = [c for c in DATE_COLS if c in df.columns]
    # Numeric: fill with column median
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    # Text: fill with 'Unknown'
    for col in text_cols:
        if df[col].isnull().any():
            df[col].fillna("Unknown", inplace=True)
    # Dates: forward fill → backward fill (Pandas 2.x compatible)
    for col in date_cols:
        if df[col].isnull().any():
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()
    after = df.isnull().sum().sum()
    print(f"[cleaning] Missing values: {before} -> {after}")
    return df
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully duplicated rows and reset index."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[cleaning] Removed {removed:,} duplicate rows "
          f"({before:,} -> {len(df):,})")
    return df
def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all date columns are proper datetime64."""
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    print(f"[cleaning] Converted date columns: {DATE_COLS}")
    return df
def standardise_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply title-case (or known transformations)."""
    for col, func in TEXT_STANDARDISE.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map(func)
    # Normalise promotion_used: anything not a real promo → 'None'
    if "promotion_used" in df.columns:
        df["promotion_used"] = df["promotion_used"].astype(str).str.strip().str.upper()
        valid_promos = {"FREESHIP", "SAVE10", "WINTER15", "SUMMER20", "NONE"}
        df["promotion_used"] = df["promotion_used"].apply(
            lambda x: x if x in valid_promos else "NONE"
        )
    print("[cleaning] Standardised text columns.")
    return df
def handle_outliers_iqr(df: pd.DataFrame,
                         cols: list = None,
                         method: str = "cap") -> pd.DataFrame:
    if cols is None:
        cols = [c for c in NUMERIC_OUTLIER_COLS if c in df.columns]
    total_flagged = 0
    for col in cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        total_flagged += outliers
        if method == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif method == "drop":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    df = df.reset_index(drop=True)
    print(f"[cleaning] Outliers handled ({method}): {total_flagged:,} values "
          f"across {len(cols)} columns.")
    return df
# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_column_names(df)
    df = convert_date_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = standardise_text_columns(df)
    df = handle_outliers_iqr(df, method="cap")
    print(f"[cleaning] Done. Final shape after cleaning: {df.shape}")
    return df
# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from modules.data_loader import load_raw_data
    raw     = load_raw_data()
    cleaned = clean_data(raw)
    print(cleaned.dtypes)
    print(cleaned.head(3).to_string())
