from pathlib import Path
import pandas as pd


def load_data(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets from CSV files.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common preprocessing for train and test data.

    - converts 'date' to datetime
    - sorts data chronologically
    - resets index
    """
    df = df.copy()

    if "date" not in df.columns:
        raise ValueError("Column 'date' was not found.")

    df["date"] = pd.to_datetime(df["date"])

    sort_columns = [
        col
        for col in ["date", "country", "store", "product"]
        if col in df.columns
    ]

    df = df.sort_values(sort_columns).reset_index(drop=True)

    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return count and percentage of missing values for each column.
    """
    missing_count = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    return pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent,
        }
    )

def drop_missing_target(
    df: pd.DataFrame,
    target_col: str = "num_sold",
) -> pd.DataFrame:
    """
    Remove rows where the target value is missing.

    Missing target values cannot be used for supervised model training
    or metric calculation. No artificial target imputation is performed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    target_col : str
        Name of the target column.

    Returns
    -------
    pd.DataFrame
        Dataframe containing only rows with known target values.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' was not found.")

    return df.dropna(subset=[target_col]).reset_index(drop=True)