from typing import Any, Union, Literal
import pandas as pd
import numpy as np
from numpy.random import randint
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
            id_name: str,
            series_names: Union[list[str], str],
            exog_names: Union[list[str], str],
            const_names: Union[list[str], str] = None,
            order: Literal["date", "multiseries"] = "date",
            y_steps: int = 1,
            return_y: bool = False
        ) -> None:
        """
        *_names: the corresponding feature column name in the dataframe 

        """
        if isinstance(series_names, str):
            series_names = [series_names]
        if isinstance(exog_names, str):
            exog_names = [exog_names]

        self.lags = lags
        self.id_name = id_name
        self.series_names = series_names
        self.exog_names = exog_names
        self.const_names = None # TODO: list type check needed
        self.var_names = None # TODO: list type check needed
        if const_names:
            self.const_names = const_names
            self.var_names = [
                name for name in self.exog_names if name not in self.const_names
            ]
        self.order = order # just for inspection, not used
        if order == "date":
            self._order = "F"
        elif order == "multiseries":
            self._order = "C"
        else:
            raise ValueError(f"Invalid `order` argument {order}")
        self.y_steps = y_steps
        self.return_y = return_y

        self.total_steps = lags + y_steps
    
    @property
    def y(self):
        return self.__y
    
    def fit(self, X, y=None):
        return self

    def _flatten(self, subframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        series = subframe[self.series_names].to_numpy()
        consts = subframe[self.const_names].head(1).to_numpy().ravel()
        vars = subframe[self.var_names].to_numpy()
        sub_X = []
        sub_y = []
        for start_idx, end_idx in enumerate(
            range(self.total_steps-1, subframe.shape[0])
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
            _fragment_vars = vars[mid_idx:end_idx + 1]

            _fragment_lags = _fragment_lags.ravel(self._order) # flatten the selected 2d array
            _fragment_vars = _fragment_vars.ravel(self._order)
            _fragment_X = np.hstack([_fragment_lags, _fragment_vars, consts])

            _fragment_y = _fragment_y.ravel(self._order)

            sub_X.append(_fragment_X)
            sub_y.append(_fragment_y)
        
        return np.vstack(sub_X), np.vstack(sub_y)
    
    def transform(self, X: pd.DataFrame, y=None):
        # NOTE: naive skipping NA -> just drop Na in `subframe`
        X = X.set_index(self.id_name, append=True).swaplevel()
        processed_X = []
        processed_y = []
        self.skipped_ids = []
        self.total_steps = self.lags + self.y_steps
        # drop Na, imputation needs to be performed beform calling this function
        X.dropna()
        for _id in X.index.levels[0].unique():
            subframe = X.loc[_id]
            if subframe.shape[0] < self.total_steps:
                self.skipped_ids.append(_id)
                continue
            _flattened_X_y = self._flatten(subframe)
            processed_X.append(_flattened_X_y[0])
            processed_y.append(_flattened_X_y[1])

        self.X_columns = ( # TODO: creation of dataframe might copy the large data
            [f"lag_{i}" for i in range(self.lags, 0, -1)] +
            [f"var_{i}" for i in range(1, self.y_steps+1)] +
            self.const_names # TODO: ensure self.const_names is list type
        )
        self.y_columns = [f"step_{i}" for i in range(1, self.y_steps+1)]
        X = np.vstack(processed_X)
        self.__y = np.vstack(processed_y)
        if self.return_y:
            return X, self.__y

        return X


class TimeSeriesTransformer3D(TimeSeriesTransformer):
    """
    Scikit-learn esitimators do not support 3 dimensional input.
    Using `TimeSeriesTransormer` to meet the requirements of scikit-learn estimator.
    Currently designed for Keras.
    """
    def _flatten(self, subframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # series = subframe[self.series_names].to_numpy()
        # consts = subframe[self.const_names].head(1).to_numpy().ravel()
        # vars = subframe[self.var_names].to_numpy()
        subarr = subframe.to_numpy()
        sub_X = []
        sub_y = []
        for start_idx, end_idx in enumerate(
            range(self.total_steps-1, subframe.shape[0])
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
            _fragment_y = subarr[mid_idx:end_idx + 1].reshape((1, self.y_steps, -1))

            sub_X.append(_fragment_X)
            sub_y.append(_fragment_y)
        
        return np.vstack(sub_X), np.vstack(sub_y)
 



def main():
    df = pd.read_csv("tests/short_raw_format.csv", index_col=1, parse_dates=[1])
    df["date"] = df.index
    df["date"] = OrdinalEncoder().fit_transform(df[["date"]])
    tst = TimeSeriesTransformer3D(
        lags=3,
        id_name="id",
        series_names="ts",
        exog_names=["exog1", "exog2", "date"],
        const_names=["exog1", "exog2"],
        y_steps=1
    )
    return tst, tst.transform(df)

if __name__ == '__main__':
    main()
