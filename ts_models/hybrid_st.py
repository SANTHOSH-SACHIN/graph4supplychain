import streamlit as st
import json
import networkx as nx
import pandas as pd
from torch_geometric.data import HeteroData
import torch
import requests
import re
import os
import numpy as np
import torch
from datetime import timedelta
import copy
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

def demand_forecasting(part_data, part_id, forecast_steps=5):
    """
    Perform demand forecasting for a specific part ID using SARIMAX model.
    
    Args:
        part_data (dict): Dictionary of DataFrames for different part IDs.
        part_id (str): Identifier for the specific part.
        forecast_steps (int, optional): Number of forecast steps. Defaults to 5.
    Returns:
        matplotlib.pyplot object, MAPE score (float)
    """
    print(f"Performing demand forecasting for Part ID: {part_id}\n")
    
    if part_id not in part_data:
        print("Error: Part ID not found.")
        return None
    
    # Retrieve the DataFrame for the selected part ID
    df = part_data[part_id].copy()
    for col in df.columns:
        if col != 'demand' and isinstance(df[col].iloc[0], (list, np.ndarray)):
            df[col] = df[col].apply(lambda x: x[0] if len(x) > 0 else np.nan)

    #Preprocess
    df.index = pd.to_datetime(df.index)
    df = df.fillna(method='ffill').fillna(method='bfill')
    y = df["demand"]  # Endogenous variable
    X = df.drop(columns=["demand"])  # Exogenous variables

    train_size = int(0.7 * len(df))
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    try:
        model = SARIMAX(
            y_train, 
            exog=X_train, 
            order=(2, 1, 2),  # ARIMA order (p,d,q)
            seasonal_order=(2, 1, 2, 6)  # Seasonal ARIMA order (P,D,Q,m)
        )
        results = model.fit(disp=False)

        
        # Forecast
        forecast = results.get_forecast(steps=len(y_test), exog=X_test)
        y_pred = forecast.predicted_mean
        mape_score = mean_absolute_percentage_error(y_test, y_pred) * 100
        plt.figure(figsize=(15, 8))
        
        # Calculate dynamic y-axis limits
        y_min = min(y_test.min(), y_pred.min())
        y_max = max(y_test.max(), y_pred.max())
        y_padding = (y_max - y_min) * 0.1

        # Plot data
        plt.ylim(y_min - y_padding, y_max + y_padding)
        plt.plot(y_train.index, y_train, label="Training Data", color="blue", alpha=0.7)
        plt.plot(y_test.index, y_test, label="Actual Demand", color="black", linewidth=2)
        plt.plot(y_test.index, y_pred, label="Forecasted Demand", color="red", linestyle="--", linewidth=2)
        plt.title(f"Demand Forecasting for Part ID: {part_id}")
        plt.xlabel("Timestamp")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt, mape_score
    
    except Exception as e:
        print(f"An error occurred during forecasting: {e}")
        return None
   
def aggregate_mape(part_data, part_id_list):
    """
    Aggregate MAPE scores for a list of part IDs.

    Args:
        part_data (dict): Dictionary of DataFrames for different part IDs.
        part_id_list (list): List of part IDs to forecast and evaluate.

    Returns:
        dict: Dictionary with part IDs as keys and their corresponding MAPE scores as values.
    """
    agg_mape = {}
    for part_id in part_id_list:
        print(f"\nProcessing Part ID: {part_id}")
        mape_score = demand_forecasting(part_data, part_id)
        if mape_score is not None:
            agg_mape[part_id] = mape_score
        else:
            agg_mape[part_id] = "Error"
    
    return agg_mape

def analyze_mape_scores(agg_mape):
    """
    Analyze aggregated MAPE scores for all parts.

    Args:
        agg_mape (dict): Dictionary with part IDs as keys and MAPE scores as values.

    Returns:
        dict: Summary statistics including mean, max, min, and standard deviation.
    """
    # Filter out entries where MAPE computation failed (e.g., "Error")
    valid_mape = {part_id: score for part_id, score in agg_mape.items() if isinstance(score, (int, float))}
    
    if not valid_mape:
        print("No valid MAPE scores found.")
        return None
    
    mape_values = list(valid_mape.values())
    mean_mape = np.mean(mape_values)
    max_mape = np.max(mape_values)
    min_mape = np.min(mape_values)
    std_mape = np.std(mape_values)
    
    # Prepare summary
    summary = {
        "Mean MAPE": mean_mape,
        "Max MAPE": max_mape,
        "Min MAPE": min_mape,
        "Standard Deviation of MAPE": std_mape,
        "Total Parts Processed": len(agg_mape),
        "Valid Parts": len(valid_mape),
        "Invalid Parts": len(agg_mape) - len(valid_mape),
    }

    return summary

def split_list_columns(df):
    """
    Splits list values in dataframe columns into separate columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns containing list values.
        
    Returns:
        pd.DataFrame: A dataframe with list columns split into separate columns.
    """
    result_df = df.copy()

    for column in df.columns:
        if column == 'demand':
            continue

        if isinstance(df[column].iloc[0], list):
            max_len = max(len(row) for row in df[column])
            for i in range(max_len):
                result_df[f"{column}_{i + 1}"] = df[column].apply(lambda x: x[i] if i < len(x) else None)
            result_df.drop(columns=[column], inplace=True)
    
    return result_df

def aggregated_demand_forecasting(part_data, part_id):
    """
    Perform aggregated demand forecasting for a specific part ID using SARIMAX model.
    
    Args:
        part_data (DataFrame): DataFrame containing demand data for parts.
        part_id (str): Identifier for the specific part.
    
    Returns:
        matplotlib.pyplot object, MAPE score (float)
    """
    # Preprocess
    part_df = part_data.copy()
    part_df = split_list_columns(part_df)
    part_df.index = pd.to_datetime(part_df.index)
    part_df = part_df.fillna(method='ffill').fillna(method='bfill')
    
    y = part_df["demand"]  # Endogenous variable
    X = part_df.drop(columns=["demand"])  # Exogenous variables
    
    train_size = int(0.7 * len(part_df))
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    try:
        model = SARIMAX(
            y_train, 
            exog=X_train, 
            order=(2, 1, 2),  # ARIMA order (p,d,q)
            seasonal_order=(2, 1, 2, 6)  # Seasonal ARIMA order (P,D,Q,m)
        )
        results = model.fit(disp=False)
        
        # Forecast
        forecast = results.get_forecast(steps=len(y_test), exog=X_test)
        y_pred = forecast.predicted_mean

        mape_score = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Plot
        plt.figure(figsize=(15, 8))
        y_min = min(y_test.min(), y_pred.min(), y_train.min())
        y_max = max(y_test.max(), y_pred.max(), y_train.max())
        y_padding = (y_max - y_min) * 0.1
        plt.ylim(y_min - y_padding, y_max + y_padding)
        plt.plot(y_train.index, y_train, label="Training Data", color="blue", alpha=0.7)
        
        plt.plot(y_test.index, y_test, label="Actual Demand", color="black", linewidth=2)
        plt.plot(y_test.index, y_pred, label="Forecasted Demand", color="red", linestyle="--", linewidth=2)
        
        plt.title(f"Aggregated Demand Forecasting for Part ID: {part_id}")
        plt.xlabel("Timestamp")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt, mape_score
    
    except Exception as e:
        print(f"An error occurred during forecasting: {e}")
        return None


def final_mape_aggregate(part_id_list:list,parser:object):
    agg_mape = {}
    for part_id in part_id_list:
        print(f"\nProcessing Part ID: {part_id}")
        
        try:
            part_data = parser.aggregate_part_features(part_id)
            forecasting_plot,mape_score = aggregated_demand_forecasting(part_data, part_id)
            agg_mape[part_id] = mape_score
            
        except Exception as e:
            print(f"Error processing Part ID {part_id}: {str(e)}")
            agg_mape[part_id] = "Error"
            
    return agg_mape 


