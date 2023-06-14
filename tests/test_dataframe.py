"""
Test the submodule `tstide.dataframe`
"""
import numpy as np
import pandas as pd
import pytest

from tstide.dataframe import mask_rows, unique_values


class Test_mask_rows:
    @pytest.fixture()
    def df(self):
        data = {
            0: (0, 1, 2, 3, 4),
            1: (10, 11, 12, 13, 14),
        }
        df = pd.DataFrame(data)
        return df

    def test_mask(self, df):
        mask = mask_rows(df, {0: (0, 4), 1: (11, 13)})
        assert (mask == np.array([True, True, False, True, True])).all()

    def test_mask_exclude(self, df):
        mask = mask_rows(df, {0: (0, 4), 1: (11, 13)}, exclude=True)
        assert (mask == ~np.array([True, True, False, True, True])).all()

    def test_return_df(self, df):
        mask = mask_rows(df, {0: (0, 4), 1: (11, 13)})
        df_masked = mask_rows(df, {0: (0, 4), 1: (11, 13)}, return_df=True)
        pd.testing.assert_frame_equal(df[mask], df_masked)

class Test_unique_values:
    @pytest.fixture()
    def df(self):
        np.random.seed(42)
        indices = np.random.randint(10, size=5)
        df = pd.DataFrame({i: np.random.randint(10, size=3) for i in indices})
        return df

    def test_unique_values(self, df):
        num_unique = unique_values(df)
        pd.testing.assert_series_equal(
            num_unique,
            pd.Series([3, 3, 2, 3], index=[6, 3, 7, 4])
        )
