import json
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns

warnings.filterwarnings('ignore')

class SingleStepARIMA:
    def __init__(self, metadata_path: str):
        """Initialize the analyzer with metadata"""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_timestamps = self.metadata['total_timestamps']
        self.base_date = self.metadata['base_date']
    
    def calculate_mape(self, actual: np.array, predicted: np.array) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(actual, predicted) * 100
    
    def train_test_split(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the time series data into training and testing sets"""
        n = len(df)
        train_size = int(n * train_size)
        return df.iloc[:train_size], df.iloc[train_size:]
    
    # ... Other methods identical to the existing ones for single-step ARIMA ...
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
        
        # Plot training data
        plt.plot(train_series.index, train_series.values, 
                label='Training Data', marker='o', linestyle='-', linewidth=2)
        
        # Plot test data
        plt.plot(test_series.index, test_series.values, 
                label='Test Data', marker='o', linestyle='-', linewidth=2)
        
        # Plot forecast
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
        
    def forecast_node(self, train_series: pd.Series, test_series: pd.Series, 
                      order: Tuple[int, int, int]=(1, 1, 1)) -> Tuple[pd.Series, float]:
        """Fit ARIMA model on training data and forecast for test period"""
        model = ARIMA(train_series, order=order)
        results = model.fit()
        forecast = results.forecast(steps=len(test_series))
        mape = self.calculate_mape(test_series.values, forecast.values)
        return forecast, mape

    # ... Rest of the single-step forecasting methods ...

class MultiStepARIMA:
    def __init__(self, metadata_path: str):
        """Initialize the analyzer with metadata"""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_timestamps = self.metadata['total_timestamps']
        self.base_date = self.metadata['base_date']
    
    def calculate_mape(self, actual: np.array, predicted: np.array) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(actual, predicted) * 100
    
    def train_test_split(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the time series data into training and testing sets"""
        n = len(df)
        train_size = int(n * train_size)
        return df.iloc[:train_size], df.iloc[train_size:]
    
    # ... Other methods identical to the existing ones for multi-step ARIMA ...
    def plot_multistep_forecast_comparison(self, 
                                         train_series: pd.Series,
                                         test_series: pd.Series,
                                         forecasts: np.ndarray, 
                                         forecast_horizons: List[int],
                                         title: str,
                                         metric: str,
                                         node_id: str,
                                         mapes: Dict[int, float],
                                         save_path: str = None):
        """Plot actual vs multi-step forecasted values"""
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        plt.plot(train_series.index, train_series.values, 
                label='Training Data', marker='o', linestyle='-', linewidth=2)
        
        # Plot test data
        plt.plot(test_series.index, test_series.values, 
                label='Test Data', marker='o', linestyle='-', linewidth=2)
        
        # Plot forecasts for each horizon
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
    
    
    def forecast_node_multistep(self, train_series: pd.Series, test_series: pd.Series,
                                forecast_horizons: List[int], order: Tuple[int, int, int]=(1, 1, 1)) -> Tuple[List[np.ndarray], Dict[int, float]]:
        """Fit ARIMA model and generate multi-step forecasts for different horizons"""
        model = ARIMA(train_series, order=order)
        results = model.fit()
        
        forecasts = []
        mapes = {}
        for horizon in forecast_horizons:
            horizon_forecast = np.zeros(len(test_series) - horizon + 1)
            for i in range(len(horizon_forecast)):
                history = pd.concat([train_series, test_series.iloc[:i]])
                model = ARIMA(history, order=order)
                results = model.fit()
                forecast = results.forecast(steps=horizon)
                horizon_forecast[i] = forecast.iloc[-1]
            
            forecasts.append(horizon_forecast)
            actual = test_series.iloc[horizon-1:].values
            mapes[horizon] = self.calculate_mape(actual, horizon_forecast)
        
        return forecasts, mapes

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
            

    # ... Rest of the multi-step forecasting methods ...
