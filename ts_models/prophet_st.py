import pandas as pd
import numpy as np
from prophet import Prophet
import json
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dateutil.relativedelta import relativedelta
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
warnings.filterwarnings('ignore')

def calculate_prophet_aic_bic(model, train_df):
    """
    Calculate AIC and BIC for a Prophet model
    
    Parameters:
    -----------
    model : Prophet model
        Fitted Prophet model
    train_df : pd.DataFrame
        Training dataframe in Prophet format with 'ds' and 'y' columns
    
    Returns:
    --------
    tuple: (AIC score, BIC score)
    """
    # Number of observations
    n = len(train_df)
    
    # Calculate log-likelihood 
    # For Prophet, we'll use the residuals to approximate log-likelihood
    forecast = model.predict(train_df)
    residuals = train_df['y'].values - forecast['yhat'].values
    
    # Estimate variance of residuals
    sigma_squared = np.var(residuals)
    
    # Number of parameters 
    # This is an approximation as Prophet has multiple components
    num_params = (
        1  # Baseline trend
        + (1 if model.yearly_seasonality else 0)  # Yearly seasonality 
        + (1 if model.weekly_seasonality else 0)  # Weekly seasonality
        + (1 if model.daily_seasonality else 0)   # Daily seasonality
        + 1  # Changepoint prior scale
        + 1  # Seasonality prior scale
    )
    
    # AIC Calculation
    # AIC = 2k - 2ln(L)
    # k is number of parameters, L is maximum likelihood
    aic = 2 * num_params + n * np.log(sigma_squared)
    
    # BIC Calculation
    # BIC = k * ln(n) - 2ln(L)
    bic = num_params * np.log(n) + n * np.log(sigma_squared)
    
    return aic, bic

class SingleStepProphet:
    def __init__(self, metadata_path: str):
        """Initialize the forecaster with metadata"""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_timestamps = self.metadata['total_timestamps']
        self.base_date = self.metadata['base_date']
    
    def calculate_mape(self, actual: np.array, predicted: np.array) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(actual, predicted) * 100
    
    def calculate_aic_bic(self, train_series: pd.Series, params: Dict = None):
        """
        Calculate and write AIC and BIC for the forecast_node method
        
        Parameters:
        -----------
        train_series : pd.Series
            Training time series data
        params : Dict, optional
            Prophet model parameters
        """
        if params is None:
            params = {
                'yearly_seasonality': False,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'seasonality_mode': 'multiplicative',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10
            }
        
        # Prepare data in Prophet format
        train_df = self.prepare_prophet_data(train_series)
        
        # Initialize and fit Prophet model
        model = Prophet(**params)
        model.fit(train_df)
        
        # Calculate AIC and BIC
        aic, bic = calculate_prophet_aic_bic(model, train_df)
        
        # Write results using Streamlit
        st.write("### AIC and BIC Scores")
        st.write(f"AIC Score: {aic:.2f}")
        st.write(f"BIC Score: {bic:.2f}")

    def load_timestamp_data(self, timestamp_files: List[str]) -> pd.DataFrame:
        """Load and combine all timestamp data files into a single DataFrame"""
        all_data = []
        
        for file_path in timestamp_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.append(data)
        
        metrics_data = []
        
        for timestamp_idx, timestamp_data in enumerate(all_data):
            date = pd.to_datetime(self.base_date) + pd.DateOffset(months=timestamp_idx)
            
            if 'node_values' in timestamp_data and 'PRODUCT_OFFERING' in timestamp_data['node_values']:
                for node in timestamp_data['node_values']['PRODUCT_OFFERING']:
                    metrics_data.append({
                        'date': date,
                        'node_id': node[4],
                        'cost': float(node[2]),
                        'demand': float(node[3])
                    })
        
        if not metrics_data:
            raise ValueError("No product offering data found in timestamp files")
            
        df = pd.DataFrame(metrics_data)
        df = df.pivot(index='date', columns='node_id', values=['cost', 'demand'])
        df = df.sort_index()
        return df

    def prepare_time_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare cost and demand DataFrames for forecasting"""
        cost_df = df['cost'].copy()
        demand_df = df['demand'].copy()
        
        cost_df = cost_df.fillna(method='ffill').fillna(method='bfill')
        demand_df = demand_df.fillna(method='ffill').fillna(method='bfill')
        
        return cost_df, demand_df

    def prepare_prophet_data(self, series: pd.Series) -> pd.DataFrame:
        """Prepare data in Prophet format"""
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        return df.sort_values('ds')

    def train_test_split(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the time series data into training and testing sets"""
        n = len(df)
        train_size = int(n * train_size)
        return df.iloc[:train_size], df.iloc[train_size:]

    def forecast_node(self, train_series: pd.Series, test_series: pd.Series, 
                     params: Dict = None) -> Tuple[pd.Series, float]:
        """Fit Prophet model on training data and forecast for test period"""
        if params is None:
            params = {
                'yearly_seasonality': False,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'seasonality_mode': 'multiplicative',
                'changepoint_prior_scale': 0.05,  # Flexibility of trend
                'seasonality_prior_scale': 10     # Flexibility of seasonality
            }
        
        # Prepare data in Prophet format
        train_df = self.prepare_prophet_data(train_series)
        
        # Initialize and fit Prophet model
        model = Prophet(**params)
        model.fit(train_df)
        
        # Create future dataframe for prediction
        future = pd.DataFrame({'ds': test_series.index})
        forecast = model.predict(future)
        
        predictions = pd.Series(forecast['yhat'].values, index=test_series.index)
        
        # Calculate MAPE
        mape = self.calculate_mape(test_series.values, predictions.values)
        self.calculate_aic_bic(train_series)

        return predictions, mape

    def plot_forecast_comparison(self, 
                               train_series: pd.Series,
                               test_series: pd.Series,
                               forecast: pd.Series, 
                               title: str,
                               metric: str,
                               node_id: str,
                               mape: float,
                               save_path: str = None):
        """Plot actual vs forecasted values with train-test split"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(train_series.index, train_series.values, 
                label='Training Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, test_series.values, 
                label='Test Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, forecast, 
                label='Forecast', marker='s', linestyle='--', linewidth=2)
        
        plt.title(f'{title}\nTest Set MAPE: {mape:.2f}%')
        plt.xlabel('Date')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/{node_id}_{metric}_forecast.png')
        else:
            plt.show()
        
        plt.close()

    def plot_mape_heatmap(self, mape_results: Dict[str, Dict[str, float]], save_path: str = None):
        """Create a heatmap of MAPE scores for all nodes and metrics"""
        metrics = ['cost', 'demand']
        nodes = list(mape_results.keys())
        
        mape_matrix = np.zeros((len(nodes), len(metrics)))
        for i, node in enumerate(nodes):
            for j, metric in enumerate(metrics):
                mape_matrix[i, j] = mape_results[node][metric]
        
        plt.figure(figsize=(10, len(nodes) * 0.5))
        sns.heatmap(mape_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   xticklabels=metrics,
                   yticklabels=nodes,
                   cmap='YlOrRd')
        
        plt.title('Test Set MAPE Scores by Node and Metric (%)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/mape_heatmap.png')
        else:
            plt.show()
        
        plt.close()

class MultiStepProphet:
    def __init__(self, metadata_path: str):
        """Initialize the forecaster with metadata"""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_timestamps = self.metadata['total_timestamps']
        self.base_date = self.metadata['base_date']
    
    def calculate_mape(self, actual: np.array, predicted: np.array) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(actual, predicted) * 100

    def calculate_aic_bic(self, train_series: pd.Series, forecast_horizons: List[int], params: Dict = None):
        """
        Calculate and write AIC and BIC for multi-step forecasting
        
        Parameters:
        -----------
        train_series : pd.Series
            Training time series data
        forecast_horizons : List[int]
            List of forecast horizons to calculate AIC/BIC for
        params : Dict, optional
            Prophet model parameters
        """
        if params is None:
            params = {
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'seasonality_mode': 'multiplicative'
            }
        
        st.write("### Multi-Step AIC and BIC Scores")
        
        for horizon in forecast_horizons:
            # Prepare data in Prophet format
            train_df = self.prepare_prophet_data(train_series)
            
            # Train model
            model = Prophet(**params)
            model.fit(train_df)
            
            # Calculate AIC and BIC
            aic, bic = calculate_prophet_aic_bic(model, train_df)
            
            # Write results for each horizon
            st.write(f"Forecast Horizon {horizon}:")
            st.write(f"AIC Score: {aic:.2f}")
            st.write(f"BIC Score: {bic:.2f}")


    def load_timestamp_data(self, timestamp_files: List[str]) -> pd.DataFrame:
        """Load and combine all timestamp data files into a single DataFrame"""
        all_data = []
        
        for file_path in timestamp_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.append(data)
        
        metrics_data = []
        
        for timestamp_idx, timestamp_data in enumerate(all_data):
            date = pd.to_datetime(self.base_date) + pd.DateOffset(months=timestamp_idx)
            
            if 'node_values' in timestamp_data and 'PRODUCT_OFFERING' in timestamp_data['node_values']:
                for node in timestamp_data['node_values']['PRODUCT_OFFERING']:
                    metrics_data.append({
                        'date': date,
                        'node_id': node[4],
                        'cost': float(node[2]),
                        'demand': float(node[3])
                    })
        
        if not metrics_data:
            raise ValueError("No product offering data found in timestamp files")
            
        df = pd.DataFrame(metrics_data)
        df = df.pivot(index='date', columns='node_id', values=['cost', 'demand'])
        return df

    def prepare_time_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare cost and demand DataFrames for forecasting"""
        cost_df = df['cost'].copy()
        demand_df = df['demand'].copy()
        
        cost_df = cost_df.fillna(method='ffill').fillna(method='bfill')
        demand_df = demand_df.fillna(method='ffill').fillna(method='bfill')
        
        return cost_df, demand_df

    def prepare_prophet_data(self, series: pd.Series) -> pd.DataFrame:
        """Prepare data in Prophet format"""
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        return df.sort_values('ds')

    def train_test_split(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the time series data into training and testing sets"""
        n = len(df)
        train_size = int(n * train_size)
        return df.iloc[:train_size], df.iloc[train_size:]

    def forecast_node_multistep(self, train_series: pd.Series, test_series: pd.Series,
                              forecast_horizons: List[int], params: Dict = None) -> Tuple[List[np.ndarray], Dict[int, float]]:
        """Fit Prophet model and generate multi-step forecasts for different horizons"""
        if params is None:
            params = {
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'seasonality_mode': 'multiplicative'
            }
        
        forecasts = []
        mapes = {}
        
        for horizon in forecast_horizons:
            horizon_forecast = np.zeros(len(test_series) - horizon + 1)
            
            for i in range(len(horizon_forecast)):
                # Prepare data up to current point
                history = pd.concat([train_series, test_series.iloc[:i]])
                train_df = self.prepare_prophet_data(history)
                
                # Train model
                model = Prophet(**params)
                model.fit(train_df)
                
                # Create future dataframe for prediction
                future = model.make_future_dataframe(periods=horizon, freq='M')
                forecast = model.predict(future)
                
                # Get prediction for the horizon
                horizon_forecast[i] = forecast.iloc[-1]['yhat']
            
            forecasts.append(horizon_forecast)
            actual = test_series.iloc[horizon-1:].values
            mapes[horizon] = self.calculate_mape(actual, horizon_forecast)
        self.calculate_aic_bic(train_series, forecast_horizons=[1, 2, 3])
        return forecasts, mapes

    def plot_multistep_forecast_comparison(self, 
                                         train_series: pd.Series,
                                         test_series: pd.Series,
                                         forecasts: List[np.ndarray], 
                                         forecast_horizons: List[int],
                                         title: str,
                                         metric: str,
                                         node_id: str,
                                         mapes: Dict[int, float],
                                         save_path: str = None):
        """Plot actual vs multi-step forecasted values"""
        plt.figure(figsize=(15, 8))
        
        plt.plot(train_series.index, train_series.values, 
                label='Training Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, test_series.values, 
                label='Test Data', marker='o', linestyle='-', linewidth=2)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(forecast_horizons)))
        for forecast, horizon, color in zip(forecasts, forecast_horizons, colors):
            plt.plot(test_series.index[horizon-1:], forecast, 
                    label=f'{horizon}-Step Forecast (MAPE: {mapes[horizon]:.2f}%)',
                    marker='s', linestyle='--', linewidth=2, color=color)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/{node_id}_{metric}_multistep_forecast.png')
        else:
            plt.show()
        
        plt.close()

    def plot_mape_heatmap_multistep(self, 
                                   mape_results: Dict[str, Dict[str, Dict[int, float]]], 
                                   forecast_horizons: List[int],
                                   save_path: str = None):
        """Create a heatmap of MAPE scores for all nodes, metrics, and forecast horizons"""
        metrics = ['cost', 'demand']
        nodes = list(mape_results.keys())
        
        # Create separate heatmaps for each forecast horizon
        for horizon in forecast_horizons:
            mape_matrix = np.zeros((len(nodes), len(metrics)))
            for i, node in enumerate(nodes):
                for j, metric in enumerate(metrics):
                    mape_matrix[i, j] = mape_results[node][metric][horizon]
            
            plt.figure(figsize=(10, len(nodes) * 0.5))
            sns.heatmap(mape_matrix, 
                       annot=True, 
                       fmt='.2f', 
                       xticklabels=metrics,
                       yticklabels=nodes,
                       cmap='YlOrRd')
            
            plt.title(f'{horizon}-Step Forecast MAPE Scores by Node and Metric (%)')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f'{save_path}/mape_heatmap_{horizon}step.png')
            else:
                plt.show()
            
            plt.close()