
import pandas as pd


def temporal_train_val_split(
    df: pd.DataFrame,
    val_start_date: str = "2016-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data into training and validation sets.

    All observations before val_start_date are used for training.
    All observations from val_start_date onward are used for validation.

    The split is performed by date to prevent future information
    from leaking into the training set.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'date' column.

    val_start_date : str
        First date included in the validation set.

    Returns
    -------
    train_df : pd.DataFrame
        Training observations.

    val_df : pd.DataFrame
        Validation observations.
    """
    if "date" not in df.columns:
        raise ValueError("Column 'date' was not found.")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    val_start_date = pd.Timestamp(val_start_date)

    train_df = df[df["date"] < val_start_date].copy()
    val_df = df[df["date"] >= val_start_date].copy()

    if train_df.empty:
        raise ValueError("Training set is empty.")

    if val_df.empty:
        raise ValueError("Validation set is empty.")

    # Safety check: training data must be strictly earlier
    # than validation data.
    if train_df["date"].max() >= val_df["date"].min():
        raise ValueError("Temporal leakage detected.")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
    )

def expanding_window_splits(
    df: pd.DataFrame,
    validation_years=(2014, 2015, 2016),
):
    """
    Generate expanding-window train/validation splits.

    For each validation year, all observations from earlier years
    are used for training, while observations from the selected year
    are used for validation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'date' column.

    validation_years : iterable of int
        Years used as validation periods.

    Yields
    ------
    train_df : pd.DataFrame
        Training observations before the validation year.

    val_df : pd.DataFrame
        Observations from the validation year.
    """
    if "date" not in df.columns:
        raise ValueError("Column 'date' was not found.")

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    for year in validation_years:
        val_start = pd.Timestamp(f"{year}-01-01")
        val_end = pd.Timestamp(f"{year + 1}-01-01")

        train_df = df[df["date"] < val_start].copy()

        val_df = df[
            (df["date"] >= val_start)
            & (df["date"] < val_end)
        ].copy()

        if train_df.empty:
            raise ValueError(
                f"Training set is empty for validation year {year}."
            )

        if val_df.empty:
            raise ValueError(
                f"Validation set is empty for validation year {year}."
            )

        if train_df["date"].max() >= val_df["date"].min():
            raise ValueError(
                f"Temporal leakage detected for validation year {year}."
            )

        yield (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
        )