"""Feature engineering for inventory observations.

All derived fields are computed from the observation itself and, for demand-change
and rolling sales, from earlier rows within the same store/product history.
"""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
    "Price", "Discount", "Weather Condition", "Holiday/Promotion",
    "Competitor Pricing", "Seasonality",
]

NUMERIC_COLUMNS = [
    "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
    "Price", "Discount", "Holiday/Promotion", "Competitor Pricing",
]

FEATURE_COLUMNS = [
    "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
    "Price", "Discount", "Holiday/Promotion", "Competitor Pricing",
    "Stock_to_Sales_Ratio", "Days_of_Inventory", "Order_to_Sales_Ratio",
    "Demand_Change", "Rolling_Sales_Average", "Forecast_Gap",
    "Price_Gap", "Inventory_Value",
]


def validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def clean_inventory_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types and invalid values without deleting potential anomalies."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "Date" in out:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Keep rows with valid observation identity; median-impute numeric values so
    # uploaded files remain usable while preserving unusual valid values.
    out = out.drop_duplicates().copy()
    for col in NUMERIC_COLUMNS:
        if col in out:
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(out[col].median())
    for col in ["Store ID", "Product ID", "Category", "Region", "Weather Condition", "Seasonality"]:
        if col in out:
            out[col] = out[col].fillna("Unknown").astype(str)
    out["Date"] = out["Date"].fillna(pd.Timestamp("2022-01-01"))
    return out.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_inventory_data(df)
    out = out.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    sales = out["Units Sold"].clip(lower=0)
    safe_sales = sales.replace(0, np.nan)
    out["Stock_to_Sales_Ratio"] = (out["Inventory Level"] / safe_sales).replace([np.inf, -np.inf], np.nan).fillna(out["Inventory Level"])
    out["Days_of_Inventory"] = out["Stock_to_Sales_Ratio"].clip(0, 3650)
    out["Order_to_Sales_Ratio"] = (out["Units Ordered"] / safe_sales).replace([np.inf, -np.inf], np.nan).fillna(out["Units Ordered"])
    out["Demand_Change"] = out.groupby(["Store ID", "Product ID"], sort=False)["Units Sold"].diff().fillna(0)
    out["Rolling_Sales_Average"] = out.groupby(["Store ID", "Product ID"], sort=False)["Units Sold"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).fillna(out["Units Sold"])
    out["Forecast_Gap"] = out["Units Sold"] - out["Demand Forecast"]
    out["Price_Gap"] = out["Price"] - out["Competitor Pricing"]
    out["Inventory_Value"] = out["Inventory Level"] * out["Price"]
    return out
