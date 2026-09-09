import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features derived from the date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'date' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional calendar features.
    """
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

    return df
