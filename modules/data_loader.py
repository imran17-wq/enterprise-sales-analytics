"""
data_loader.py
==============
Handles loading of the Excel sales dataset into a Pandas DataFrame.

Author : Sales Analytics Dashboard
Purpose: Portfolio / Internship project
"""

import os
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
import os
DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "Product-Sales-Region.xlsx")
SHEET_NAME = 0          # first sheet
EXPECTED_MIN_ROWS = 100  # sanity-check threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(filepath: str = DEFAULT_FILE) -> pd.DataFrame:
    """
    Load the Excel sales file into a raw Pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .xlsx file.

    Returns
    -------
    pd.DataFrame
        Unprocessed data exactly as it appears in the workbook.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the loaded data appears suspiciously small.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Sales data file not found: '{filepath}'. "
            "Please place the Excel file in the project root directory."
        )

    df = pd.read_excel(filepath, sheet_name=SHEET_NAME)

    if len(df) < EXPECTED_MIN_ROWS:
        raise ValueError(
            f"Loaded data has only {len(df)} rows. "
            f"Expected at least {EXPECTED_MIN_ROWS}. "
            "Please check the Excel file."
        )

    print(f"[data_loader] Loaded {len(df):,} rows x {len(df.columns)} columns "
          f"from '{filepath}'")
    return df


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of column metadata (dtype, null count, unique count).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    info = pd.DataFrame({
        "dtype":        df.dtypes.astype(str),
        "null_count":   df.isnull().sum(),
        "null_pct":     (df.isnull().sum() / len(df) * 100).round(2),
        "unique_count": df.nunique(),
    })
    return info


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_raw_data()
    print(raw.head(3).to_string())
    print("\nColumn info:\n", get_column_info(raw).to_string())
