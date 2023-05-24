from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from skforecast.utils import initialize_lags


SERIES_PATH = "short_series.csv"
series = pd.read_csv(SERIES_PATH, parse_dates=[0])

# y 是 _create_lags 的 paramter
y = series['1'].to_numpy() # np.array([11., 21., 31., 41., nan, nan, nan, nan, nan])

###########################################
# 以下參數會交給物件的參數設定，並非傳入函式的參數
lags = initialize_lags("ForecasterAutoregMultiSeries", 2) # 2 是傳入物件的 lags
# ^^ = np.array([1, 2])
max_lag = max(lags)
# ^^^^^ = 2
###########################################

def _create_lags(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # TODO:請你把有 nan 的地方去掉, 像是 52, 53 行那裡的答案 (請腦補那些 0 是 nan, 我打錯了)
    # 這邊是原始的 code 讓你參考他怎麼做
    n_splits = len(y) - max_lag
    if n_splits <= 0:
        raise ValueError(
            f'The maximum lag ({max_lag}) must be less than the length '
            f'of the series ({len(y)}).'
        )

    X_data = np.full(shape=(n_splits, len(lags)), fill_value=np.nan, dtype=float)

    for i, lag in enumerate(lags):
        X_data[:, i] = y[max_lag - lag: -lag]

    y_data = y[max_lag:]
    return X_data, y_data 
    # X_data -> array([[21., 11.],     y_data -> array([31., 41.,  0.,  0.,  0.,  0.,  0.]))
    #                  [31., 21.],
    #                  [41., 31.],
    #                  [ 0., 41.],
    #                  [ 0.,  0.],
    #                  [ 0.,  0.],
    #                  [ 0.,  0.]])


@pytest.mark.parametrize("y", [y])
def test_create_lags(y: np.ndarray):
    X_data, y_data = _create_lags(y)
    np.testing.assert_array_equal(X_data, np.array([[21., 11.], [31., 21.]])) # Answer for X_data
    np.testing.assert_equal(y_data, np.array([31, 41])) # Answer for y_dtat



# 想再做加上 exogenous_variable 的，但這個和 skforecast 物件無關
# recall: series 在第 15 行
EXOG_PATH = "short_exog.csv"
exog = pd.read_csv(EXOG_PATH)
# id  exog1  exog2
#  0    100    200
#  1    110    210
#  2    120    220

strict_ans = pd.read_csv("answer_strict_X_y.csv", index_col=0, parse_dates=[5])
ans = pd.read_csv("answer_X_y.csv", index_col=0, parse_dates=[5])

def create_X_y(
    series: pd.DataFrame,
    exog: pd.DataFrame,
    lags: int,
    strict=True
) -> Tuple[np.ndarray, np.ndarray]: # 當然可以用 DataFrame, Series 做, 最後記得 .to_numpy() 就好
    # TODO:
    # strict=True 完全不包含 NaN
    # strict=False 可以包含 NaN，但最少要有一個值
    # 要符合 answer_X_y.csv answer_strict_X_y.csv (執行下面的 test 就知道有沒有了)
    # 我打算把 searchDate 當作一個 feature, 做不出來可以先跳過
    pass


@pytest.mark.parametrize(
    "series, exog, lags, strict, answer",
    [
        (series, exog, 2, True, strict_ans),
        (series, exog, 2, False, ans)
    ]
)
def test_create_X_y(
    series: pd.DataFrame,
    exog: pd.DataFrame,
    lags: int,
    strict: bool,
    answer: pd.DataFrame
):
    X, y = create_X_y(series, exog, lags, strict)
    np.testing.assert_array_equal(
        X[:, :4], answer[["lag_1", "lag_2", "exog1", "exog2"]].to_numpy()
    )
    np.testing.assert_array_equal(y, answer["y"].to_numpy())


@pytest.mark.parametrize(
    "series, exog, lags, strict, answer",
    [
        (series, exog, 2, True, strict_ans),
        (series, exog, 2, False, ans)
    ],
)
def test_create_X_y_with_date(
    series: pd.DataFrame,
    exog: pd.DataFrame,
    lags: int,
    strict: bool,
    answer: pd.DataFrame
):
    X, y = create_X_y(series, exog, lags, strict)
    np.testing.assert_array_equal(
        X, answer[["lag_1", "lag_2", "exog1", "exog2", "date"]].to_numpy()
    )
    np.testing.assert_array_equal(y, answer["y"].to_numpy())
