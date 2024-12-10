# Timeseries Forecasting

This page of the application allows users to select a node from the Temporal Heterogeneous Graph and analyze the demand data associated with that node using various time series forecasting models.

## Data Source Selection

The user can select the data source for the analysis from the following options:

*   Local Directory: Select this option if you have already downloaded the data and it is stored in a local directory.
*   Server: Select this option if you want to fetch the data from a remote server.

## Upload Metadata File

The user needs to upload the metadata.json file associated with the Temporal Heterogeneous Graph. This file contains information about the nodes and edges in the graph.

## Select Node for Analysis

The user can select a node from the Temporal Heterogeneous Graph for analysis. The demand data associated with the selected node is displayed in the form of a line chart.

## Select Forecasting Model

The user can select a forecasting model from the following options:

*   ARIMA
*   SARIMA
*   XGBoost
*   Prophet

## Select Forecasting Type

The user can select the type of forecasting from the following options:

*   Single-Step Forecasting: This option is used to forecast the demand for a single time step ahead.
*   Multi-Step Forecasting: This option is used to forecast the demand for multiple time steps ahead.

## Select Forecast Horizons (Steps)

The user can select the forecast horizons (steps) for the multi-step forecasting. The user can select one or more horizons from the following options:

*   1
*   2
*   3
*   4
*   5
*   6

## Training Parameters

The user can adjust the training parameters for the XGBoost model. The parameters are:

*   Lookback Period: This parameter determines how many time steps to look back in the data to train the model.
*   Number of Estimators: This parameter determines how many trees to train.
*   Max Depth: This parameter determines the maximum depth of each tree.
*   Learning Rate: This parameter determines the learning rate of the model.

## View Results

Once the user has selected the node, forecasting model, forecasting type, and forecast horizons, the results of the forecasting are displayed in the form of a line chart. The chart shows the actual demand data, the forecasted data, and the mean absolute percentage error (MAPE) for each forecast horizon.
