from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..dataframe import valid_folds
from ..transformer import TimeSeriesTransformer


def unravel(
    array: np.ndarray,
    shape: tuple[int, int],
    order: Literal["C", "F"]
):
    if order == "F":
        return array.reshape(shape[::-1]).T
    elif order == "C":
        return array.reshape(shape)
    else:
        raise ValueError(f"The order must be `C` or `F`. {order} is invalid.")


def create_future_df(
    series: pd.DataFrame,
    n_steps: int,
    ts_trans: TimeSeriesTransformer,
    var_future: np.ndarray,
):
    df_future = pd.DataFrame(
        np.zeros([n_steps, len(series.columns)], dtype=np.float32),
        columns=series.columns,
        index=pd.date_range(series.index[-1], periods=n_steps+1)[1:]
    )
    df_future[ts_trans.const_names] = series[ts_trans.const_names].tail(1).to_numpy().ravel()
    df_future[ts_trans.var_names] = var_future
    return df_future


def rolling_predict(
    regressor: RandomForestRegressor,
    series: pd.DataFrame,
    n_steps: int,
    ts_trans: TimeSeriesTransformer,
    stdscaler: StandardScaler
):
    # TODO: DEBUG this function
    df_future = series.iloc[-(ts_trans.lags+n_steps):].copy()
    pred_array = np.zeros([n_steps+ts_trans.y_steps-1, len(ts_trans.series_names)])
    pred_count = np.zeros(n_steps+ts_trans.y_steps-1).reshape(-1, 1)

    series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
    # time_step_left = n_steps
    for start_idx in range(n_steps-ts_trans.y_steps+1):
        # mid_idx = start_idx + ts_trans.lags
        end_idx = start_idx + ts_trans.lags + ts_trans.y_steps

        X = ts_trans.rearrange_row(df_future.iloc[start_idx:end_idx])

        if stdscaler:
            X = stdscaler.transform(X)

        y_pred = regressor.predict(X)
        y_arr = unravel(y_pred, (ts_trans.y_steps, len(series_indexes)), ts_trans._order)

        rolling_end = start_idx + ts_trans.y_steps
        pred_array[start_idx:rolling_end] = (
            pred_array[start_idx:rolling_end] * pred_count[start_idx:rolling_end] + y_arr
        )
        pred_count[start_idx:rolling_end] += 1
        pred_array[start_idx:rolling_end] /= pred_count[start_idx:rolling_end]

        # if ts_trans.y_steps > time_step_left:
        #     for i in range(time_step_left):
        #         df_future.iloc[mid_idx+i, series_indexes] = y_arr[i]
        #     break

        # time_step_left -= ts_trans.y_steps
        df_future.iloc[-n_steps:, series_indexes] = pred_array[:n_steps]


    return df_future

def recursive_predict(
    regressor: RandomForestRegressor, # TODO: Not restrict to RandomForest
    series: pd.DataFrame,
    n_steps: int,
    ts_trans: TimeSeriesTransformer,
    stdscaler: StandardScaler,
):
    """
    - series: The dataframe includes futre n_steps dates.
    The number of rows must > lags + n_steps and exog variables must be filled.
    """

    # df_included = series.iloc[-ts_trans.lags:]
    # df_included.index = pd.to_datetime(df_included.index)
    # df_included.asfreq(pd.infer_freq(df_included.index))

    # df_future = create_future_df(series, n_steps, ts_trans, var_future)
    # df_total = pd.concat([df_included, df_future])
    df_future = series.iloc[-(ts_trans.lags+n_steps):].copy()

    series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
    time_step_left = n_steps
    for start_idx in range(0, n_steps, ts_trans.y_steps):
        mid_idx = start_idx + ts_trans.lags
        end_idx = start_idx + ts_trans.lags + ts_trans.y_steps

        # X, y = ts_trans.rearrange_X_y(df_future.iloc[start_idx:end_idx])
        X = ts_trans.rearrange_row(df_future.iloc[start_idx:end_idx])

        if stdscaler:
            X = stdscaler.transform(X)

        y_pred = regressor.predict(X)
        y_arr = unravel(y_pred, (ts_trans.y_steps, len(series_indexes)), ts_trans._order)

        if ts_trans.y_steps > time_step_left:
            for i in range(time_step_left):
                df_future.iloc[mid_idx+i, series_indexes] = y_arr[i]
            break

        time_step_left -= ts_trans.y_steps
        df_future.iloc[mid_idx:end_idx, series_indexes] = y_arr

    return df_future


def forecast(
    reg: RandomForestRegressor,
    trailing_data: pd.DataFrame,
    ts_trans: TimeSeriesTransformer,
    n_steps: int = 7,
    method: Literal["recursive", "rolling"] = "recursive",
    decimals: Optional[tuple[int]] = (2, 0), # TODO: Change (This settings is only designed for our project)
    stdscaler: Optional[StandardScaler] = None
):
    if method == "recursive":
        method = recursive_predict
    elif method == "rolling":
        method = rolling_predict
    else:
        raise ValueError("pred_func only support 'recursive' and 'rolling'")
    y_pred_arr = np.zeros(
        [len(ts_trans.series_names), trailing_data.index.get_level_values(0).unique().shape[0], n_steps],
        dtype=np.float32
    )
    for row, _id in enumerate(trailing_data.index.get_level_values(0).unique()):
        pred_frame = method(reg, trailing_data.loc[_id], n_steps, ts_trans, stdscaler).iloc[-n_steps:]
        for index, (feature_name, _decimal) in enumerate(zip(ts_trans.series_names, decimals)):
            y_pred_arr[index, row] = np.around(pred_frame[feature_name].to_numpy().T, decimals=_decimal)

    return y_pred_arr


def backtesting(
    reg: RandomForestRegressor,
    series: pd.DataFrame,
    ts_trans: TimeSeriesTransformer,
    n_folds: int = 5,
    n_steps: int = 7,
    method: Literal["recursive", "rolling"] = "recursive",
    batch_normalization: bool = True,
    return_cache: bool = False,
    _dataset_cache: Optional[dict] = {},
):
    if return_cache:
        return _dataset_cache

    caching = False
    lags_steps = (ts_trans.lags, ts_trans.y_steps)
    y_pred_list = []
    for fold_n, valid_fold in enumerate(
        valid_folds(
            series,
            pd.to_datetime(series.index.get_level_values(1)),
            n_folds,
            n_steps
        )
    ):
        if (ts_trans.lags, ts_trans.y_steps) not in _dataset_cache:
            caching = True
            _dataset_cache[lags_steps] = {
                "X_train": [],
                "y_train": [],
                "trailing_data": []
            }
        if caching:
            X_train, y_train, trailing_data = ts_trans.train_test_split(valid_fold, n_steps)
            _dataset_cache[lags_steps]["X_train"].append(X_train)
            _dataset_cache[lags_steps]["y_train"].append(y_train)
            _dataset_cache[lags_steps]["trailing_data"].append(trailing_data)
        else:
            X_train = _dataset_cache[lags_steps]["X_train"][fold_n]
            y_train = _dataset_cache[lags_steps]["y_train"][fold_n]
            trailing_data = _dataset_cache[lags_steps]["trailing_data"][fold_n]

        if batch_normalization:
            stdscaler = StandardScaler()
            X_train = stdscaler.fit_transform(stdscaler)

        reg.fit(X_train, y_train)
        y_pred = forecast(reg, trailing_data, ts_trans, n_steps, method, stdscaler=stdscaler)
        y_pred_list.append(y_pred)

    return y_pred_list
