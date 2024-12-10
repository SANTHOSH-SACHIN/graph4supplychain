# Hybrid Model User Guide

This page of the app is used to forecast demand using a hybrid model. The hybrid model is a combination of a graph modelled data and a time series forecasting model. The Graph Dataframe composes of Aggregated/Non Aggregated columns information making iut account for graph nature of the data, and the time series forecasting model is used to forecast the demand for each node.

## Data Source

The first step is to select the data source. The data source can be either a local directory or a server. If you select a local directory, you will need to enter the path to the directory. If you select a server, you will need to enter the URL of the server.


## Model Choice

The next step is to select the type of model you want to use. The options are "Non-Aggregated Columns" and "Aggregated Columns". If you select "Non-Aggregated Columns", the model will be trained on each column of the data separately. If you select "Aggregated Columns", the model will be trained on the aggregated data.

## Part ID

The next step is to select the part ID. The part ID is the column in the data that contains the demand information. You can select the part ID from the list of columns in the data.

## Aggregation Method

If you selected "Aggregated Columns" in the previous step, you will need to select the aggregation method. The options are "mean", "sum", "min", and "max". The aggregation method is used to aggregate the data across the columns.

## Run Forecasting

Once you have selected the part ID and the aggregation method, you can run the forecasting model. The model will be trained on the data and the results will be displayed in a plot.

## Plot

The plot shows the forecasted demand for each node in the graph. The x-axis shows the time steps, and the y-axis shows the demand. The plot also shows the actual demand for each node.

## MAPE

The MAPE (Mean Absolute Percentage Error) is a measure of the accuracy of the forecast. The MAPE is calculated for each node in the graph and is displayed in the plot.

## Results

The results of the forecasting model are displayed in the plot. The results include the forecasted demand for each node, the actual demand for each node, and the MAPE for each node.
