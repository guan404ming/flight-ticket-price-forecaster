from typing import Literal

import keras
import numpy as np
import pandas as pd

from ..transformer import TimeSeriesTransformer3D


def recursive_predict_keras(
    model: keras.Sequential,
    series: np.ndarray,
    n_steps: int,
    ts_trans: TimeSeriesTransformer3D,
):
    """
    - series: The dataframe includes futre n_steps dates.
    The number of rows must > lags + n_steps and exog variables must be filled.
    """
    raise NotImplementedError

    df_included = series.iloc[-(ts_trans.lags+n_steps):].copy()

    series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
    start_idx = 0
    mid_idx = ts_trans.lags
    end_idx = ts_trans.lags + ts_trans.y_steps
    time_step_left = n_steps
    for _ in range(0, n_steps, ts_trans.y_steps):
        X = df_included.iloc[start_idx:mid_idx]
        y_pred = model.predict(X)
        y_arr = unravel(y_pred, (ts_trans.y_steps, len(series_indexes)), ts_trans._order)

        if ts_trans.y_steps > time_step_left:
            for i in range(time_step_left):
                df_included.iloc[mid_idx+i, series_indexes] = y_arr[i]
            break

        df_included.iloc[mid_idx:end_idx, series_indexes] = y_arr

        time_step_left -= ts_trans.y_steps
        start_idx += ts_trans.y_steps
        mid_idx += ts_trans.y_steps
        end_idx += ts_trans.y_steps

    return df_included.iloc[-n_steps:, series_indexes]


def main():
    df = pd.read_csv("ticket_info_test.csv")
    df.index = df["searchDate"]
    # ts_trans = TimeSeriesTransformer(
    #     7,
    #     "legId",
    #     "totalFare",
    #     ["searchDate", "segmentsDepartureTimeRaw", "segmentsAirlineCode"],
    #     ["segmentsDepartureTimeRaw", "segmentsAirlineCode"],
    #     y_steps=2
    # )
    # # X = ts_trans.transform(df)
    # # y = ts_trans.y

    # ticket = df[df["legId"] == "00d56034d4d9e5611543da4fd818fa4f"]
    # predict_days = 8
    # new_df = pd.DataFrame(np.zeros([predict_days, df.shape[1]]))
    # new_df.columns = df.columns
    # new_df[["legId", "segmentsDepartureTimeRaw", "segmentsAirlineCode"]] = ticket[["legId", "segmentsDepartureTimeRaw", "segmentsAirlineCode"]].iloc[-1]

    # last_day = pd.to_datetime(ticket["searchDate"][-1])
    # for i in range(predict_days):
    #     new_df.at[i, "searchDate"] = last_day + pd.Timedelta(days=i)

    # sandbox = pd.concat([ticket, new_df])
    # # y_pred = recursive_predict(
    # #     sandbox,
    # #     predict_days,
    # #     ts_trans
    # # )
    # # print(y_pred)


if __name__ == '__main__':
    main()
