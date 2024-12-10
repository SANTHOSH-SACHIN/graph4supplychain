import json
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
warnings.filterwarnings('ignore')
import streamlit as st
class BaseSarimaAnalyzer:
    """Base class with common functionality for SARIMA analysis"""
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

    def find_optimal_sarima_params(self, 
                                 train_series: pd.Series,
                                 seasonal_period: int = 12) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """Find optimal SARIMA parameters using grid search"""
        best_aic = float('inf')
        best_params = None
        best_seasonal_params = None
        
        p_params = range(0, 3)
        d_params = range(0, 2)
        q_params = range(0, 3)
        P_params = range(0, 2)
        D_params = range(0, 2)
        Q_params = range(0, 2)
        
        for p in p_params:
            for d in d_params:
                for q in q_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                try:
                                    model = SARIMAX(
                                        train_series,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonal_period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    results = model.fit(disp=False)
                                    # AIC BIC
                                    st.write(f" AIC: {results.aic:.2f}")
                                    st.write(f" BIC: {results.bic:.2f}")
                                    if results.aic < best_aic:
                                        best_aic = results.aic
                                        best_params = (p, d, q)
                                        best_seasonal_params = (P, D, Q, seasonal_period)
                                except:
                                    continue
        
        return best_params, best_seasonal_params

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

class SingleStepSarimaAnalyzer(BaseSarimaAnalyzer):
    """Analyzer for single-step SARIMA forecasting"""
    
    def forecast_node_sarima(self, 
                           train_series: pd.Series, 
                           test_series: pd.Series,
                           seasonal_period: int = 12) -> Tuple[np.ndarray, float, Tuple, Tuple]:
        """Fit SARIMA model and generate single-step forecasts"""
        order_params, seasonal_params = self.find_optimal_sarima_params(train_series, seasonal_period)
        forecasts = np.zeros(len(test_series))
        
        for i in range(len(test_series)):
            history = pd.concat([train_series, test_series.iloc[:i]])
            model = SARIMAX(
                history,
                order=order_params,
                seasonal_order=seasonal_params,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            st.write(f" AIC: {results.aic:.2f}")
            st.write(f" BIC: {results.bic:.2f}")
            forecast = results.forecast(steps=1)
            forecasts[i] = forecast.iloc[0]
        
        mape = self.calculate_mape(test_series.values, forecasts)
        return forecasts, mape, order_params, seasonal_params

    def plot_forecast_comparison(self, 
                               train_series: pd.Series,
                               test_series: pd.Series,
                               forecast: np.ndarray, 
                               title: str,
                               metric: str,
                               node_id: str,
                               mape: float,
                               save_path: str = None):
        """Plot actual vs forecasted values"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(train_series.index, train_series.values, 
                label='Training Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, test_series.values, 
                label='Test Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, forecast, 
                label=f'Forecast (MAPE: {mape:.2f}%)',
                marker='s', linestyle='--', linewidth=2)
        
        plt.title(f'{title}\nSARIMA Model')
        plt.xlabel('Date')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/{node_id}_{metric}_sarima_forecast.png')
            plt.close()
            return f'{save_path}/{node_id}_{metric}_sarima_forecast.png'
        else:
            plt.close()
            return plt
    
    def plot_mape_heatmap(self, 
                         mape_results: Dict[str, Dict[str, float]], 
                         save_path: str = None):
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
        
        plt.title('SARIMA Single-Step Forecast MAPE Scores (%)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/sarima_single_step_mape_heatmap.png')
            plt.close()
            return f'{save_path}/sarima_single_step_mape_heatmap.png'
        else:
            plt.close()
            return plt

class MultiStepSarimaAnalyzer(BaseSarimaAnalyzer):
    """Analyzer for multi-step SARIMA forecasting"""
    
    def forecast_node_multistep_sarima(self, 
                                     train_series: pd.Series, 
                                     test_series: pd.Series,
                                     forecast_horizons: List[int],
                                     seasonal_period: int = 12) -> Tuple[List[np.ndarray], Dict[int, float], Tuple, Tuple]:
        """Fit SARIMA model and generate multi-step forecasts for different horizons"""
        order_params, seasonal_params = self.find_optimal_sarima_params(train_series, seasonal_period)
        
        forecasts = []
        mapes = {}
        
        for horizon in forecast_horizons:
            horizon_forecast = np.zeros(len(test_series) - horizon + 1)
            
            for i in range(len(horizon_forecast)):
                history = pd.concat([train_series, test_series.iloc[:i]])
                model = SARIMAX(
                    history,
                    order=order_params,
                    seasonal_order=seasonal_params,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                st.write(f" AIC: {results.aic:.2f}")
                st.write(f" BIC: {results.bic:.2f}")
                forecast = results.forecast(steps=horizon)
                horizon_forecast[i] = forecast.iloc[-1]
            
            forecasts.append(horizon_forecast)
            actual = test_series.iloc[horizon-1:].values
            mapes[horizon] = self.calculate_mape(actual, horizon_forecast)
        
        return forecasts, mapes, order_params, seasonal_params

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
        
        plt.plot(train_series.index, train_series.values, 
                label='Training Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, test_series.values, 
                label='Test Data', marker='o', linestyle='-', linewidth=2)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(forecast_horizons)))
        for forecast, horizon, color in zip(forecasts, forecast_horizons, colors):
            plt.plot(test_series.index[horizon-1:], forecast, 
                    label=f'{horizon}-Step Forecast (MAPE: {mapes[horizon]:.2f}%)',
                    marker='s', linestyle='--', linewidth=2, color=color)
        
        plt.title(f'{title}\nSARIMA Model')
        plt.xlabel('Date')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/{node_id}_{metric}_multistep_sarima_forecast.png')
            plt.close()
            return f'{save_path}/{node_id}_{metric}_multistep_sarima_forecast.png'
        else:
            plt.close()
            return plt
    
    def plot_mape_heatmap_multistep(self, 
                                   mape_results: Dict[str, Dict[str, Dict[int, float]]], 
                                   forecast_horizons: List[int],
                                   save_path: str = None):
        """Create a heatmap of MAPE scores for all nodes, metrics, and forecast horizons"""
        metrics = ['cost', 'demand']
        nodes = list(mape_results.keys())
        heatmap_paths = []
        
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
            
            plt.title(f'{horizon}-Step SARIMA Forecast MAPE Scores (%)')
            plt.tight_layout()
            
            if save_path:
                path = f'{save_path}/sarima_mape_heatmap_{horizon}step.png'
                plt.savefig(path)
                heatmap_paths.append(path)
            plt.close()
        
        return heatmap_paths if save_path else None