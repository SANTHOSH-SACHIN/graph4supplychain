import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import streamlit as st
warnings.filterwarnings('ignore')

class XGBoostForecaster:
    def __init__(self, metadata_path: str):
        """Initialize the forecaster with metadata"""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_timestamps = self.metadata['total_timestamps']
        self.base_date = self.metadata['base_date']
        self.scaler = StandardScaler()
        
    def calculate_mape(self, actual: np.array, predicted: np.array) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(actual, predicted) * 100
    
    def create_features(self, series: pd.Series, lags: int = 6) -> pd.DataFrame:
        """Create time series features including lags and date features"""
        df = pd.DataFrame(index=series.index)
        df['y'] = series

        # Ensure the index is a DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Add lag features
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = series.shift(lag)

        # Add date features
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Add rolling statistics
        for window in [3, 6]:
            df[f'rolling_mean_{window}'] = series.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = series.rolling(window=window).std()

        # Drop rows with NaN values
        df = df.dropna()
        return df
    
    def train_test_split(self, series: pd.Series, train_size: float = 0.7) -> Tuple[pd.Series, pd.Series]:
        """Split the time series into training and testing sets"""
        n = int(len(series) * train_size)
        train_series = series.iloc[:n]
        test_series = series.iloc[n:]
        return train_series, test_series
        
    def prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix X and target vector y"""
        y = df['y'].values
        X = df.drop('y', axis=1).values
        return X, y
    
    def single_step_forecast(self, train_series: pd.Series, test_series: pd.Series,
                           params: Dict = None) -> Tuple[np.ndarray, float, float, float]:
        """Perform single-step forecasting using XGBoost"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'learning_rate': 0.01,
                'max_depth': 5,
                'min_child_weight': 1
            }
            
        # Create features for train and test sets
        train_df = self.create_features(train_series)
        test_df = self.create_features(pd.concat([train_series, test_series]))
        test_df = test_df.iloc[len(train_df):]
        
        # Prepare data
        X_train, y_train = self.prepare_xy(train_df)
        X_test, y_test = self.prepare_xy(test_df)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        mape = self.calculate_mape(y_test, predictions)
        
        # Calculate AIC and BIC
        residuals = y_test - predictions
        n = len(y_test)
        k = model.get_params()['n_estimators'] * (model.get_params()['max_depth'] + 1)  # Approximate number of parameters
        sigma = np.std(residuals)
        log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - (np.sum(residuals**2)) / (2 * sigma**2)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        st.write(f"Single Step Forecast - AIC: {aic:.2f}, BIC: {bic:.2f}")
        
        return predictions, mape
    
    def multi_step_forecast(self, train_series: pd.Series, test_series: pd.Series,
                          forecast_horizons: List[int], params: Dict = None) -> Tuple[List[np.ndarray], Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Perform multi-step forecasting using XGBoost"""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 1
            }
        
        forecasts = []
        mapes = {}
        aics = {}
        bics = {}
        
        for horizon in forecast_horizons:
            # Initialize array for storing predictions
            horizon_forecast = np.zeros(len(test_series) - horizon + 1)
            
            for i in range(len(horizon_forecast)):
                # Create features using all available data up to current point
                history = pd.concat([train_series, test_series.iloc[:i]])
                features_df = self.create_features(history)
                
                # Prepare training data
                X, y = self.prepare_xy(features_df)
                X = self.scaler.fit_transform(X)
                
                # Train model
                model = xgb.XGBRegressor(**params)
                model.fit(X, y)
                
                # Make multi-step prediction
                last_features = features_df.iloc[-1:]
                current_pred = last_features.copy()
                
                for step in range(horizon):
                    # Update features for next prediction
                    X_next = self.scaler.transform(current_pred.drop('y', axis=1))
                    pred = model.predict(X_next)[0]
                    
                    if step == horizon - 1:
                        horizon_forecast[i] = pred
                    
                    # Update current prediction DataFrame for next step
                    current_pred['y'] = pred
                    for lag in range(6, 0, -1):
                        if f'lag_{lag}' in current_pred.columns:
                            current_pred[f'lag_{lag}'] = current_pred[f'lag_{lag-1}'] if lag > 1 else pred
                    
                    # Update rolling statistics
                    for window in [3, 6]:
                        current_pred[f'rolling_mean_{window}'] = pred
                        current_pred[f'rolling_std_{window}'] = 0
            
            forecasts.append(horizon_forecast)
            actual = test_series.iloc[horizon-1:].values
            mapes[horizon] = self.calculate_mape(actual, horizon_forecast)
            
            # Calculate AIC and BIC for this horizon
            residuals = actual - horizon_forecast
            n = len(actual)
            k = model.get_params()['n_estimators'] * (model.get_params()['max_depth'] + 1)  # Approximate number of parameters
            sigma = np.std(residuals)
            log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - (np.sum(residuals**2)) / (2 * sigma**2)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            aics[horizon] = aic
            bics[horizon] = bic
            
        # Output AIC and BIC for multi-step forecasts
        st.write("Multi-Step Forecast - AIC and BIC for each horizon:")
        for horizon in forecast_horizons:
            st.write(f"Horizon {horizon} - AIC: {aics[horizon]:.2f}, BIC: {bics[horizon]:.2f}")
        
        return forecasts, mapes
    
    def plot_forecast_comparison(self, 
                               train_series: pd.Series,
                               test_series: pd.Series,
                               forecast: np.ndarray, 
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