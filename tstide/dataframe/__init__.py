"""
Helper functions for dataframe data processing.
"""
import pandas as pd

from ._inspect_values import mask_rows, unique_values


def train_set(
    df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    reserved_days: int = 42, # 7 * 6
):
    if not isinstance(date_index, pd.DatetimeIndex):
        date_index = pd.DatetimeIndex(date_index)
    split_day = date_index.max() - pd.to_timedelta(reserved_days, unit="D")
    return df[date_index <= split_day]


def valid_folds(
    df: pd.DataFrame,
    date_index: pd.DatetimeIndex,
    n_folds: int = 5,
    n_steps: int = 7
):
    if not isinstance(date_index, pd.DatetimeIndex):
        date_index = pd.DatetimeIndex(date_index)
    # each fold includes training and validation set
    # The data transformer will separate them.
    for _fold_n in range(n_folds, 0, -1):
        split_day = date_index.max() - pd.to_timedelta(n_steps * _fold_n, unit="D")
        yield df[date_index <= split_day]
