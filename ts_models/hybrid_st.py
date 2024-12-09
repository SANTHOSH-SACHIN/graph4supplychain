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
# from parser_st import TemporalHeterogeneousGraphParser


def demand_forecasting(part_data, part_id, forecast_steps=5):
    """
    Perform demand forecasting for a specific part ID using SARIMAX model
    
    Args:
        part_data (dict): Dictionary of DataFrames for different part IDs
        part_id (str): Identifier for the specific part
        forecast_steps (int, optional): Number of forecast steps. Defaults to 5.
    """
    print(f"Performing demand forecasting for Part ID: {part_id}\n")
    
    # Validate part ID
    if part_id not in part_data:
        print("Error: Part ID not found.")
        return None
    
    # Retrieve the DataFrame for the selected part ID
    df = part_data[part_id].copy()
    
    # Flatten nested columns if needed
    for col in df.columns:
        if col != 'demand' and isinstance(df[col].iloc[0], (list, np.ndarray)):
            df[col] = df[col].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    
    # Handle potential missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Separate features and target variable
    y = df["demand"]  # Endogenous variable
    X = df.drop(columns=["demand"])  # Exogenous variables
    
    # Train-test split (70:30)
    train_size = int(0.7 * len(df))
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    try:
        # Define and train the SARIMAX model
        model = SARIMAX(
            y_train, 
            exog=X_train, 
            order=(1, 1, 1),  # ARIMA order (p,d,q)
            seasonal_order=(1, 1, 1, 12)  # Seasonal ARIMA order (P,D,Q,m)
        )
        results = model.fit(disp=False)
        
        # Print model summary
        print(results.summary())
        
        # Forecast
        forecast = results.get_forecast(steps=len(y_test), exog=X_test)
        
        # Predicted values and confidence intervals
        y_pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Calculate MAPE
        mape_score = mean_absolute_percentage_error(y_test, y_pred) * 100
        print(f"\nModel MAPE: {mape_score:.2f}%")
        
        # Create the plot with error handling
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        plt.plot(y_train.index, y_train, label="Training Data", color="blue", alpha=0.7)
        
        # Plot actual test data
        plt.plot(y_test.index, y_test, label="Actual Demand", color="black", linewidth=2)
        
        # Plot forecasted demand
        plt.plot(y_test.index, y_pred, label="Forecasted Demand", color="red", linestyle="--", linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(
            y_test.index,
            conf_int.iloc[:, 0],
            conf_int.iloc[:, 1],
            color="pink",
            alpha=0.3,
            label="Confidence Interval"
        )
        
        plt.title(f"Demand Forecasting for Part ID: {part_id}")
        plt.xlabel("Timestamp")
        plt.ylabel("Demand")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent cutting off labels
        plt.tight_layout()
        
        # Return plot for further use or display
        return plt,mape_score
    
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
    
    # Extract MAPE values
    mape_values = list(valid_mape.values())
    
    # Calculate statistics
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
    
    # Print the summary
    print("\nMAPE Score Analysis:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return summary

def split_list_columns(df):
    """
    Splits list values in dataframe columns into separate columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns containing list values.
        
    Returns:
        pd.DataFrame: A dataframe with list columns split into separate columns.
    """
    result_df = df.copy()  # Start with a copy of the original dataframe

    # Iterate through each column
    for column in df.columns:
        # Ensure that 'demand' column is not mistakenly split
        if column == 'demand':
            continue  # Skip splitting for 'demand' column
        # Check if the first value in the column is a list (to identify list columns)
        if isinstance(df[column].iloc[0], list):
            # Create new columns for each element in the list
            max_len = max(len(row) for row in df[column])  # Find the max list length
            for i in range(max_len):
                result_df[f"{column}_{i + 1}"] = df[column].apply(lambda x: x[i] if i < len(x) else None)
            # Drop the original column
            result_df.drop(columns=[column], inplace=True)
    
    return result_df

def aggregated_demand_forecasting(part_data, part_id):
    part_df = part_data.copy()
    part_df = split_list_columns(part_df)
    part_df.index = pd.to_datetime(part_df.index)
    
    # Flatten nested columns if needed
    # for col in part_df.columns:
    #     if col != 'demand' and isinstance(part_df[col].iloc[0], (list, np.ndarray)):
    #         part_df[col] = part_df[col].apply(lambda x: x[0] if len(x) > 0 else np.nan)
    
    # Handle potential missing values
    part_df = part_df.fillna(method='ffill').fillna(method='bfill')
    
    # Separate features and target variable
    y = part_df["demand"]  # Endogenous variable
    X = part_df.drop(columns=["demand"])  # Exogenous variables
    # X = part_df[[part_id,"W_agg","F_agg"]]
    
    # print(X)
    # print(y)
    
    # Train-test split (70:30)
    train_size = int(0.7 * len(part_df))
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    

        # Define and train the SARIMAX model
    model = SARIMAX(
            y_train, 
            exog=X_train, 
            order=(1, 1, 1),  # ARIMA order (p,d,q)
            seasonal_order=(1, 1, 1, 12)  # Seasonal ARIMA order (P,D,Q,m)
        )
    results = model.fit(disp=False)
        
        # Print model summary
    print(results.summary())
        
        # Forecast
    forecast = results.get_forecast(steps=len(y_test), exog=X_test)
        
        # Predicted values and confidence intervals
    y_pred = forecast.predicted_mean
    conf_int = forecast.conf_int()
        
        # Calculate MAPE
    mape_score = mean_absolute_percentage_error(y_test, y_pred) * 100
    print(f"\nModel MAPE: {mape_score:.2f}%")
        
        # Create the plot with error handling
    plt.figure(figsize=(15, 8))
        
        # Plot training data
    plt.plot(y_train.index, y_train, label="Training Data", color="blue", alpha=0.7)
        
        # Plot actual test data
    plt.plot(y_test.index, y_test, label="Actual Demand", color="black", linewidth=2)
        
        # Plot forecasted demand
    plt.plot(y_test.index, y_pred, label="Forecasted Demand", color="red", linestyle="--", linewidth=2)
        
    # Plot confidence intervals
    plt.fill_between(
        y_test.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color="pink",
        alpha=0.3,
        label="Confidence Interval"
    )
    
    plt.title(f"Demand Forecasting for Part ID: {part_id}")
    plt.xlabel("Timestamp")
    plt.ylabel("Demand")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Return plot for further use or display
    return plt,mape_score

def final_mape_aggregate(part_id_list:list,parser:object):
    agg_mape = {}
    for part_id in part_id_list:
        print(f"\nProcessing Part ID: {part_id}")
        
        try:
            # Retrieve part data
            part_data = parser.aggregate_part_features(part_id)
            
            # Run demand forecasting for the part and capture the plot
            forecasting_plot,mape_score = aggregated_demand_forecasting(part_data, part_id)
            
            # Extract MAPE score from the printed result in `aggregated_demand_forecasting`
            # Ensure that the function returns MAPE as a variable too, if needed
            # mape_score = forecasting_plot  # Assuming it returns a MAPE score as part of the result
            
            # Add MAPE score to the dictionary
            agg_mape[part_id] = mape_score
        except Exception as e:
            print(f"Error processing Part ID {part_id}: {str(e)}")
            agg_mape[part_id] = "Error"
            
    return agg_mape 


