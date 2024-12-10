# Quick Start Guide

Get started with Graph4SupplyChain in minutes! This guide will walk you through basic usage and key features.

## Time series forecasting

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   The application will open in your default browser.

2. **Run Time Series Forecasting**
   - Select a forecasting model:
     - ARIMA
     - Prophet
     - XGBoost
     - SARIMA
   - Choose forecast horizon
   - Configure model parameters
   - Click "Generate Forecast"


3. **Generate Forecasts**
   - Select nodes to forecast
   - Choose ARIMA model
   - Set forecast period to 30 days
   - View and export results

## GNN based forecasting

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   The application will open in your default browser.

2. **Run GNN Forecasting**
   - Upload a metadata.json file
   - Select a forecasting Type:
     - Single step
     - Multi step
   - Configure your model preferences
   - Customize your Hyperparameters
   - Click "Train Model"

3. **Download the Trained Model**
   - Download the trained model as a .pth file

4. **Dashboard for metrics**
   - View and analyze performance metrics
   - Loss and Metrics per epoch is being displayed as a plot