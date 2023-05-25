"""
test the submodule `tstide.transformer`
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder
from tstide.transformer import TimeSeriesTransformer, TimeSeriesTransformer3D
from pathlib import Path


@pytest.fixture()
def time_series():
    df = pd.read_csv(Path(__file__).parent.joinpath("short_raw_format.csv"))
    df.index = df["date"]
    df["date"] = OrdinalEncoder().fit_transform(df[["date"]])
    return df


def test_TimeSeriesTransformer(time_series: pd.DataFrame):
    """
    There should be more tests here but I was tired.
    """
    tst = TimeSeriesTransformer(
        lags=3,
        id_name="id",
        series_names=["ts1", "ts2"],
        exog_names=["exog1", "exog2", "date"],
        const_names=["exog1", "exog2"],
        y_steps=2
    )
    X = tst.transform(time_series)
    np.testing.assert_array_equal(
        X,
        np.array([[ 42.,  52.,  62., -42., -52., -62.,   6.,   7., 110., 210.]])
    )
    np.testing.assert_array_equal(
        tst.y,
        np.array([[ 72,  82, -72, -82]])
    )
    # Those id with no enough days to form X and y are neglected
    # id 1 and 3 both have only 3 days
    # The transformer need 3 + 2 = 5 days (sum of lags and y_steps)
    assert tst.skipped_ids == [1, 3]


def test_TimeSeriesTransformer(time_series: pd.DataFrame):
    tst = TimeSeriesTransformer3D(
        lags=3,
        id_name="id",
        series_names=["ts1", "ts2"],
        y_steps=2
    )
    X, y = tst.transform(time_series)

    assert tst.X_columns == ["date", "ts1", "ts2", "exog1", "exog2"]
    np.testing.assert_array_equal(
        X,
        np.array([
            [
                [  3.,  42., -42., 110., 210.],
                [  4.,  52., -52., 110., 210.],
                [  5.,  62., -62., 110., 210.],
            ]
        ])
    )
    assert tst.y_columns == ["ts1", "ts2"]
    np.testing.assert_array_equal(
        y,
        np.array([
            [
                [ 72, -72],
                [ 82, -82],
            ],
        ])
    )
    # Those id with no enough days to form X and y are neglected
    # id 1 and 3 both have only 3 days
    # The transformer need 3 + 2 = 5 days (sum of lags and y_steps)
    assert tst.skipped_ids == [1, 3]
