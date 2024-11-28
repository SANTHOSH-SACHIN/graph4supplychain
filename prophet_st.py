import pandas as pd
import numpy as np
from prophet import Prophet
import json
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_percentage_error
warnings.filterwarnings('ignore')

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
                'seasonality_mode': 'additive'
            }
        
        # Prepare data in Prophet format
        train_df = self.prepare_prophet_data(train_series)
        
        # Initialize and fit Prophet model
        model = Prophet(**params)
        model.fit(train_df)
        
        # Create future dataframe for prediction
        # future = model.make_future_dataframe(periods=len(test_series), freq='M')
        future = pd.DataFrame({'ds': test_series.index})
        forecast = model.predict(future)
        
        # Get predictions for test set
        # predictions = pd.Series(
        #     forecast.iloc[-len(test_series):]['yhat'].values,
        #     index=test_series.index
        # )

        predictions = pd.Series(forecast['yhat'].values, index=test_series.index)
        
        # Calculate MAPE
        mape = self.calculate_mape(test_series.values, predictions.values)
        
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