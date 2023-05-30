from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

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


# def rolling_predict(
#     regressor: RandomForestRegressor,
#     series: pd.DataFrame,
#     n_steps: int,
#     ts_trans: TimeSeriesTransformer,
#     var_future: np.ndarray
# ):
#     df_included = series.iloc[-(ts_trans.lags+n_steps):]
#     date = df_included.index[-1]
#     pred_array = np.zeros([n_steps, len(ts_trans.series_names)])
#     pred_count = np.zeros(n_steps).reshape(n_steps, -1)

#     series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
#     time_step_left = n_steps
#     for start_idx in range(0, n_steps):
#         mid_idx = start_idx + ts_trans.lags
#         end_idx = start_idx + ts_trans.lags + ts_trans.y_steps

#         X = ts_trans.last_window(df_included.iloc[start_idx:mid_idx])

#         y_pred = regressor.predict(X)
#         y_arr = unravel(y_pred, (ts_trans.y_steps, len(series_indexes)), ts_trans._order)

#         if ts_trans.y_steps > time_step_left:
#             for i in range(time_step_left):
#                 df_included.iloc[mid_idx+i, series_indexes] = y_arr[i]
#             break

#         time_step_left -= ts_trans.y_steps
#         df_included.iloc[mid_idx:end_idx, series_indexes] = y_arr

#     return df_included.iloc[-n_steps:, series_indexes]
    


def recursive_predict(
    regressor: RandomForestRegressor, # TODO: Not restrict to RandomForest
    series: pd.DataFrame,
    n_steps: int,
    ts_trans: TimeSeriesTransformer,
    var_future: np.ndarray,
):
    """
    - series: The dataframe includes futre n_steps dates.
    The number of rows must > lags + n_steps and exog variables must be filled.
    """

    df_included = series.iloc[-ts_trans.lags:]
    df_included.index = pd.to_datetime(df_included.index)
    df_included.asfreq(pd.infer_freq(df_included.index))

    df_future = create_future_df(series, n_steps, ts_trans, var_future)
    df_total = pd.concat([df_included, df_future])
    
    series_indexes = [series.columns.get_loc(name) for name in ts_trans.series_names]
    time_step_left = n_steps
    for start_idx in range(0, n_steps, ts_trans.y_steps):
        mid_idx = start_idx + ts_trans.lags
        end_idx = start_idx + ts_trans.lags + ts_trans.y_steps

        X, y = ts_trans.rearrange_X_y(df_total.iloc[start_idx:end_idx])

        y_pred = regressor.predict(X)
        y_arr = unravel(y_pred, (ts_trans.y_steps, len(series_indexes)), ts_trans._order)

        if ts_trans.y_steps > time_step_left:
            for i in range(time_step_left):
                df_total.iloc[mid_idx+i, series_indexes] = y_arr[i]
            break

        time_step_left -= ts_trans.y_steps
        df_total.iloc[mid_idx:end_idx, series_indexes] = y_arr


    return df_total.iloc[-n_steps:]
    # mid_idx = ts_trans.lags
    # end_idx = ts_trans.lags + ts_trans.y_steps
    # time_step_left = n_steps
    # for _ in range(0, n_steps, ts_trans.y_steps):
    #     X = ts_trans.create_last_window(df_included.iloc[start_idx:mid_idx])

    #     y_pred = regressor.predict(X)
    #     y_arr = unravel(y_pred, (ts_trans.y_steps, len(series_indexes)), ts_trans._order)

    #     if ts_trans.y_steps > time_step_left:
    #         for i in range(time_step_left):
    #             df_included.iloc[mid_idx+i, series_indexes] = y_arr[i]
    #         break

    #     df_included.iloc[mid_idx:end_idx, series_indexes] = y_arr

    #     time_step_left -= ts_trans.y_steps
    #     start_idx += ts_trans.y_steps
    #     mid_idx += ts_trans.y_steps
    #     end_idx += ts_trans.y_steps

    # return df_total.iloc[-n_steps:, series_indexes]


def main():
    df = pd.read_csv("ticket_info_test.csv")
    df.index = df["searchDate"]

    df_train = df.copy()

    df_train["searchDate"] = pd.to_datetime(df_train["searchDate"])
    df_train["segmentsDepartureTimeRaw"] = pd.to_datetime(df_train["segmentsDepartureTimeRaw"])

    def parse_searchDate(x: datetime):
        return (x.year - 2000) * 10000 + x.month * 100 + x.day
    def parse_depart(x: datetime):
        return ((x.year - 2000) * 10000 + x.month * 100 + x.day) * 10000 + x.hour * 100 + x.minute

    df_train["searchDate"] = df_train["searchDate"].apply(parse_searchDate)
    df_train["segmentsDepartureTimeRaw"] = df_train["segmentsDepartureTimeRaw"].apply(parse_depart)

    ordinal_enc = OrdinalEncoder()
    encode_features = ["segmentsAirlineCode"]
    df_train[encode_features] = ordinal_enc.fit_transform(df_train[encode_features])
    df_train["totalFare"] = df_train["totalFare"].astype(np.float32)
    df_train["seatsRemaining"] = df_train["seatsRemaining"].astype(np.float32)
    df_train["searchDate"] = df_train["searchDate"].astype(np.float32)
    df_train["segmentsDepartureTimeRaw"] = df_train["segmentsDepartureTimeRaw"].astype(np.float32)
    df_train["segmentsAirlineCode"] = df_train["segmentsAirlineCode"].astype(np.float32)

    ts_trans = TimeSeriesTransformer(
        7,
        "legId",
        "totalFare",
        ["searchDate", "segmentsDepartureTimeRaw", "segmentsAirlineCode"],
        ["searchDate"],
        y_steps=1
    )

    X = ts_trans.transform(df_train)
    y = ts_trans.y
    
    reg = RandomForestRegressor()
    reg.fit(X, y)

    ticket = df_train[df_train["legId"] == "00d56034d4d9e5611543da4fd818fa4f"]
    ### Create Empty dataframe to store new data
    # Recursive prediction need the exog vars in future
    # fill exogs at first
    ticket.info()
    def searchDate_generator(last_value):
        day = int(last_value % 100)
        month = int(last_value // 100 % 100)
        year = int(last_value // 10000 % 100 + 2000)
        date = datetime(year, month, day)
        while True:
            date += timedelta(days=1)
            value = parse_searchDate(date)
            yield np.array([value], dtype=np.float32)


    predict_days = 7

    _res = []
    for _, value in zip(range(predict_days), searchDate_generator(ticket.loc[ticket.index[-1], "searchDate"])):
        _res.append(value)
    _res = np.hstack(_res).reshape(7, -1)

    _frame = recursive_predict(reg, ticket[["totalFare", "seatsRemaining", "searchDate", "segmentsDepartureTimeRaw", "segmentsAirlineCode"]], predict_days, ts_trans, _res)
    print(_frame)
    print(_frame.dtypes)



    # new_df = pd.DataFrame(np.zeros([predict_days, df.shape[1]]))
    # new_df.columns = df.columns
    # new_df[["legId", "segmentsDepartureTimeRaw", "segmentsAirlineCode"]] = ticket[["legId", "segmentsDepartureTimeRaw", "segmentsAirlineCode"]].iloc[-1]

    # last_day = pd.to_datetime(ticket["searchDate"][-1])
    # for i in range(predict_days):
        # new_df.at[i, "searchDate"] = last_day + pd.Timedelta(days=i)

    # sandbox = pd.concat([ticket, new_df])
    # sandbox["searchDate"] = pd.to_datetime(sandbox["searchDate"])
    # sandbox["segmentsDepartureTimeRaw"] = pd.to_datetime(sandbox["segmentsDepartureTimeRaw"])
    # sandbox["searchDate"] = sandbox["searchDate"].apply(parse_searchDate)
    # sandbox["segmentsDepartureTimeRaw"] = sandbox["segmentsDepartureTimeRaw"].apply(parse_depart)
    # sandbox[encode_features] = ordinal_enc.transform(sandbox[encode_features])

    # y_pred = recursive_predict(
    #     reg,
    #     ticket,
    #     predict_days,
    #     ts_trans,
    #     encoded_future_dates
    # )
    # print(ticket)
    # print(y_pred)


if __name__ == '__main__':
    main()
