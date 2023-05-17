# Flight Ticket Price Forecaster

## Overview

This Python Flight Ticket Price Forecaster project aims to learn price trends from historical flight ticket prices and make predictions about future ticket prices. Flight prices are time series data, where the prices may be influenced by previous time points or even older information. The goal is to forecast the ticket prices based on historical data and various forecasting models.

The project utilizes models designed for time series analysis such as skforecast and Prophet. Additionally, it explores the use of deep learning techniques for predicting flight ticket prices.

## Dataset

The project uses the "Flight Prices Dataset" obtained from dilwong/flightprices (CC BY 4.0). This dataset consists of one-way flight prices found on Expedia between 2022-04-16 and 2022-10-05.

## Experiment: Flight Prices Data for the Current Year

An additional dataset for the current year's flight prices is available at [https://github.com/dilwong/FlightPrices](https://github.com/dilwong/FlightPrices). This dataset is regularly crawled to fetch the current ticket price data. It can be used to test the effectiveness of the prediction models and potentially improve the predictions for the current year.

## Features

The features used for predicting flight prices include:

- Time interval between the query date and the flight date.
- Flight date.
- Departure and destination airports.
- Flight duration.
- Direct flight indicator.
- Remaining seats.
- (optional) Carrier model.

The correlation between price and remaining seats is observed, as they exhibit similar patterns over time. Airlines often adjust ticket prices based on demand, and short-term trends, especially price surges before the departure date, pose a challenge for price prediction.

## System Structure and Modules

The system employs different approaches to handle the flight price prediction task. Three main modules used are:

- **Regressors** from scikit-learn: Random Forest Regressor and other regression models from scikit-learn.
- **Skforecast**: A library for time series analysis that combines scikit-learn regressors with autoregressive (AR) models.
- **Prophet**: A time series forecasting model developed by Facebook, which captures trend, seasonality, and holiday effects.

The system structure includes the following methods of handling input data:

1. **Direct Concatenation:** Multiple ticket data are directly concatenated into a large two-dimensional array, where each row represents a ticket at a specific time point. This approach allows the model to consider the influence of different tickets on each other. However, handling the large and sparse array and managing missing values (e.g., unsold or already departed tickets) can be challenging.

2. **Fixed Time Interval as Periodic Input Data:** The data is divided into fixed time intervals (lags), where each interval consists of multiple time points. The ticket data within each interval are flattened and concatenated to form a smaller array. This approach reduces the size of the array and mitigates the issue of missing values. However, the length of the lag needs to be determined, and it can impact the model's performance. Certain time points may have a higher proportion of data, resulting in imbalanced representations.

3. **Training Models per Ticket and Propagating Trends:** This approach utilizes the Prophet model, which not only predicts values but also identifies underlying trends in the dataset. Each ticket is trained separately, and the learned trends are passed to the next ticket's model. This ensures continuous and comprehensive learning of ticket variations. However, it may lead to error propagation and potentially reduce accuracy. Other models cannot utilize this approach.

The system structure and modules provide different perspectives and insights into flight price prediction, catering to various requirements and considerations.
