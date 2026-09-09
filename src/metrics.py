
import numpy as np


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE value in percent.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~np.isnan(y_true)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when y_true contains zero values.")

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100