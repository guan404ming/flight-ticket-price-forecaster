# Author: ywnien <willie3184@gmail.com>

from typing import Any, Dict, Iterable, Union

import numpy as np
import pandas as pd


def mask_rows(
        dataframe: pd.DataFrame,
        key_values: Dict[str, Union[Any, Iterable[Any]]],
        exclude: bool =False,
        return_df: bool=False
    ):
    """
    Example: 
    * `key_values`: `{"segmentsAirlineName": ('Spirit Airlines', 'Delta', 'Alaska Airlines')}`
    
    With this argument, the funciton return the mask that includes those rows of the three Airline.
    Then `dataframe[mask]` to get the filtered dataframe.
    * `exclude=True` to return the opposite mask to exclude unwated rows.
    * `return_frame=True` to return masked dataframe equibalent to `dataframe[mask]`
    """
    mask = np.zeros(dataframe.shape[0], dtype=bool)
    for key, values in key_values.items():
        if isinstance(values, str) or not isinstance(values, Iterable):
            mask |= (dataframe[key].values == values)
            continue
        for value in values:
            mask |= (dataframe[key].values == value)
    if exclude:
        mask = ~mask

    if return_df:
        return dataframe[mask]

    return mask

def unique_values(dataframe: pd.DataFrame):
    num_unique = np.zeros(dataframe.shape[1], dtype=int)
    for index, column_name in enumerate(dataframe.columns):
        num_unique[index] = dataframe[column_name].unique().size
    return pd.Series(num_unique, index=dataframe.columns)
