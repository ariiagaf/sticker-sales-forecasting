import numpy as np
import pandas as pd


REFERENCE_DATE = pd.Timestamp("2010-01-01")


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """add calendar and seasonal features from the date column"""

    if "date" not in df.columns:
        raise ValueError("Column 'date' was not found.")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["time_idx"] = (df["date"] - REFERENCE_DATE).dt.days

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    df["day_of_year_sin"] = np.sin(
        2 * np.pi * (df["day_of_year"] - 1) / 365.25
    )
    df["day_of_year_cos"] = np.cos(
        2 * np.pi * (df["day_of_year"] - 1) / 365.25
    )

    return df