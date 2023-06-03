from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    """
    Set `return_y` to true and call transform will return both `X`, `y`.
    Unfortanely, the Pipeline object only support for transforming `X`.
    Argument of `y` in methods fit and transform will be neglected.
    Getting `y` by the property `y` is recommmended.
    """
    def __init__(
            self,
            lags: int,
            # id_name: str,
            series_names: Union[list[str], str],
            exog_names: Union[list[str], str],
            var_names: Union[list[str], str] = None,
            order: Literal["date", "multiseries"] = "date",
            y_steps: int = 1,
        ) -> None:
        """
        *_names: the corresponding feature column name in the dataframe

        """
        if isinstance(series_names, str):
            series_names = [series_names]
        if isinstance(exog_names, str):
            exog_names = [exog_names]
        if isinstance(var_names, str):
            var_names = [var_names]

        self.lags = lags
        # self.id_name = id_name
        self.series_names = series_names
        self.exog_names = exog_names
        self.var_names = var_names

        if var_names:
            self.const_names = [
                name for name in self.exog_names if name not in self.var_names
            ]
        else:
            self.const_names = exog_names

        self.order = order # just for inspection, not used
        if order == "date":
            self._order = "F"
        elif order == "multiseries":
            self._order = "C"
        else:
            raise ValueError(f"Invalid `order` argument {order}")

        self.y_steps = y_steps
        # self.total_steps = lags + y_steps

    @property
    def X(self):
        return self.__X

    @property
    def y(self):
        return self.__y

    @property
    def X_columns(self) -> list[str]:
        if self.order == "date":
            series_title = [
                f"{name}_{-i}"
                for name in self.series_names
                for i in range(self.lags, 0, -1)
            ]
            return series_title + self.var_names + self.const_names
        elif self.order == "multiseries":
            series_title = [
                f"{name}_{-i}"
                for i in range(self.lags, 0, -1)
                for name in self.series_names
            ]
            return series_title + self.var_names + self.const_names

    @property
    def y_columns(self) -> list[str]:
        if self.order == "date":
            return [
                f"{name}_{i}"
                for name in self.series_names
                for i in range(1, self.y_steps+1)
            ]
        elif self.order == "multiseries":
            return [
                f"{name}_{i}"
                for i in range(1, self.y_steps+1)
                for name in self.series_names
            ]

    def rearrange_row(self, subframe: pd.DataFrame) -> np.ndarray:
        series = subframe[self.series_names].to_numpy(np.float32)
        consts = subframe[self.const_names].head(1).to_numpy(np.float32).ravel()
        vars = subframe[self.var_names].to_numpy(np.float32)

        _fragment_lags = series[:self.lags]
        _fragment_vars = vars[self.lags:self.lags + 1]

        _fragment_lags = _fragment_lags.ravel(self._order) # flatten the selected 2d array
        _fragment_vars = _fragment_vars.ravel(self._order)

        return np.hstack([_fragment_lags, _fragment_vars, consts]).reshape(1, -1)

    def rearrange_X_y(self, subframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        series = subframe[self.series_names].to_numpy(np.float32)
        consts = subframe[self.const_names].head(1).to_numpy(np.float32).ravel()
        vars = subframe[self.var_names].to_numpy(np.float32)
        sub_X = []
        sub_y = []
        for start_idx, end_idx in enumerate(
            range(self.lags + self.y_steps - 1, subframe.shape[0])
        ):
            # X range from start_idx to start_idx + self.lags - 1
            #  ->  start_idx         mid_idx = start_idx + self.lags
            #      v                 v
            #  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            #      \   self.lags   /           ^
            #                                  end_idx <-
            # y range from start_idx + self.lags to end_idx
            mid_idx = start_idx + self.lags

            _fragment_lags = series[start_idx:mid_idx]
            _fragment_y = series[mid_idx:end_idx + 1]
            # _fragment_vars = vars[mid_idx:end_idx + 1]
            _fragment_vars = vars[mid_idx:mid_idx + 1]

            _fragment_lags = _fragment_lags.ravel(self._order) # flatten the selected 2d array
            _fragment_vars = _fragment_vars.ravel(self._order)
            _fragment_X = np.hstack([_fragment_lags, _fragment_vars, consts])

            _fragment_y = _fragment_y.ravel(self._order)

            sub_X.append(_fragment_X)
            sub_y.append(_fragment_y)

        return np.vstack(sub_X), np.vstack(sub_y)

    def train_X_y(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # drop Na, imputation needs to be performed beform calling this function
        dataframe = dataframe.dropna()
        # dataframe = dataframe.set_index(self.id_name, append=True).swaplevel()
        processed_X = []
        processed_y = []
        self.skipped_ids = []
        for _id in dataframe.index.get_level_values(0).unique():
            subframe = dataframe.loc[_id]
            if subframe.shape[0] < (self.lags + self.y_steps):
                self.skipped_ids.append(_id)
                continue
            _rearranged_X_y = self.rearrange_X_y(subframe)
            processed_X.append(_rearranged_X_y[0])
            processed_y.append(_rearranged_X_y[1])

        return np.vstack(processed_X), np.vstack(processed_y)

    def train_test_split(self, dataframe: pd.DataFrame, n_steps: int = 7):
        # drop Na, imputation needs to be performed beform calling this function
        dataframe = dataframe.dropna()
        date_index = pd.to_datetime(dataframe.index.get_level_values(1))
        last_day = date_index.max()
        train_end = last_day - pd.Timedelta(n_steps, unit="D")
        last_window_start = last_day - pd.Timedelta(self.lags + n_steps, unit="D")

        train_set = dataframe[date_index <= train_end]
        X_train, y_train = self.train_X_y(train_set)

        trailing_data = dataframe[(date_index > last_window_start) & (date_index <= last_day)].copy()
        for _id in trailing_data.index.get_level_values(0).unique():
            if trailing_data.loc[_id].shape[0] < self.lags + n_steps:
                trailing_data.drop(index=_id, level=0, inplace=True)

        return X_train, y_train, trailing_data


    def extract_y_true(self, trailing_data: pd.DataFrame, n_steps: int = 7):
        y_true_arr = np.zeros(
            [
                len(self.series_names),
                trailing_data.index.get_level_values(0).unique().shape[0],
                n_steps
            ],
            dtype=np.float32
        )
        for row, _id in enumerate(trailing_data.index.get_level_values(0).unique()):
            for index, feature_name in enumerate(self.series_names):
                y_true_arr[index, row] = (
                    trailing_data
                    .loc[_id, feature_name]
                    .iloc[-n_steps:]
                    .to_numpy()
                    .T
                )

        return y_true_arr


    def transform(self, X: pd.DataFrame, y=None):
        """
        Return X only. Access `y` by `TimeSeriesTransformer.y`
        """
        X, y = self.train_X_y(X)
        self.__X = X
        self.__y = y
        return X

    def fit(self, X, y=None):
        return self


class TimeSeriesTransformer3D(TimeSeriesTransformer):
    """
    Scikit-learn esitimators do not support 3 dimensional input.
    Using `TimeSeriesTransormer` to meet the requirements of scikit-learn estimator.
    Currently designed for Keras.

    NOTE: I think I should write this first and then let TimeSeriesTransformer
     inherit it ... Too late now.
    """
    def __init__(
            self,
            lags: int,
            id_name: str,
            series_names: Union[list[str], str],
            y_steps: int = 1,
        ) -> None:
        if isinstance(series_names, str):
            series_names = [series_names]

        self.lags = lags
        self.id_name = id_name
        self.series_names = series_names
        self.y_steps = y_steps

        # self.total_steps = lags + y_steps

    @property
    def X_columns(self) -> list[str]:
        return self.__X_columns

    @property
    def y_columns(self) -> list[str]:
        return self.series_names

    def rearrange_X_y(self, subframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        series = subframe[self.series_names].to_numpy()
        subarr = subframe.to_numpy()
        sub_X = []
        sub_y = []
        for start_idx, end_idx in enumerate(
            range(self.lags + self.y_steps - 1, subframe.shape[0])
        ):
            # X range from start_idx to start_idx + self.lags - 1
            #  ->  start_idx         mid_idx = start_idx + self.lags
            #      v                 v
            #  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            #      \   self.lags   /           ^
            #                                  end_idx <-
            # y range from start_idx + self.lags to end_idx
            mid_idx = start_idx + self.lags

            _fragment_X = subarr[start_idx:mid_idx].reshape((1, self.lags, -1))
            _fragment_y = series[mid_idx:end_idx + 1].reshape((1, self.y_steps, -1))

            sub_X.append(_fragment_X)
            sub_y.append(_fragment_y)

        return np.vstack(sub_X), np.vstack(sub_y)

    def transform(self, X: pd.DataFrame, y=None) -> tuple[np.ndarray, np.ndarray]:
        self.__X_columns = [name for name in X.columns if name != self.id_name]
        return self.train_X_y(X)


def main():
    df = pd.read_csv("tests/short_raw_format.csv", index_col=1, parse_dates=[1])
    df["date"] = df.index
    df["date"] = OrdinalEncoder().fit_transform(df[["date"]])
    tst = TimeSeriesTransformer3D(
        lags=3,
        id_name="id",
        series_names=["ts1", "ts2"],
        # exog_names=["exog1", "exog2", "date"],
        # const_names=["exog1", "exog2"],
        # order="multiseries",
        y_steps=2
    )
    tst.transform(df)

    print(tst.X_columns, end="\n\n")
    print(tst.X, end="\n\n")
    print(tst.y_columns, end="\n\n")
    print(tst.y, end="\n\n")
    print(tst.skipped_ids, end="\n\n")
    return tst, tst.transform(df)

if __name__ == '__main__':
    main()
