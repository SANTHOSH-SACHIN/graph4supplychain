import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

class SingleStepCNNLSTMAnalyzer:
    def __init__(self, metadata_path: str):
        """Initialize the analyzer with metadata"""
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_timestamps = self.metadata['total_timestamps']
        self.base_date = self.metadata['base_date']
        self.scalers = {}
    
    # [Previous methods remain the same, just moved to this file]
    def calculate_mape(self, actual: np.array, predicted: np.array) -> float:
        return mean_absolute_percentage_error(actual, predicted) * 100
    
    def create_sequences(self, data: np.array, lookback: int) -> Tuple[np.array, np.array]:
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def train_test_split(self, df: pd.DataFrame, train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n = len(df)
        train_size = int(n * train_size)
        return df.iloc[:train_size], df.iloc[train_size:]
    
    def build_cnn_lstm_model(self, lookback: int) -> Sequential:
        model = Sequential([
            Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(lookback, 1)),
            Conv1D(filters=32, kernel_size=2, activation='relu'),
            LSTM(50, activation='relu', return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def plot_forecast_comparison(self, train_series, test_series, predictions, title, metric, node_id, mape, lookback, save_path=None):
        plt.figure(figsize=(15, 8))
        plt.plot(train_series.index, train_series.values, label='Training Data', marker='o', linestyle='-', linewidth=2)
        plt.plot(test_series.index, test_series.values, label='Test Data', marker='o', linestyle='-', linewidth=2)
        prediction_index = test_series.index[lookback:]
        plt.plot(prediction_index, predictions, label=f'CNN-LSTM Forecast (MAPE: {mape:.2f}%)', marker='s', linestyle='--', linewidth=2)
        plt.title(f'{title}\nCNN-LSTM Model')
        plt.xlabel('Date')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/{node_id}_{metric}_cnn_lstm_forecast.png')
        return plt
    
    def plot_mape_heatmap(self, mape_results: Dict[str, Dict[str, float]], save_path: str = None):
        metrics = ['cost', 'demand']
        nodes = list(mape_results.keys())
        mape_matrix = np.zeros((len(nodes), len(metrics)))
        for i, node in enumerate(nodes):
            for j, metric in enumerate(metrics):
                mape_matrix[i, j] = mape_results[node][metric]
        
        plt.figure(figsize=(10, len(nodes) * 0.5))
        sns.heatmap(mape_matrix, annot=True, fmt='.2f', xticklabels=metrics,
                    yticklabels=nodes, cmap='YlOrRd')
        plt.title('CNN-LSTM Forecast MAPE Scores (%)')
        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/cnn_lstm_mape_heatmap.png')
        return plt
    
    def load_timestamp_data(self, timestamp_files: List[str]) -> pd.DataFrame:
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
        cost_df = df['cost'].copy()
        demand_df = df['demand'].copy()
        cost_df = cost_df.fillna(method='ffill').fillna(method='bfill')
        demand_df = demand_df.fillna(method='ffill').fillna(method='bfill')
        return cost_df, demand_df
    
    def forecast_node_cnn_lstm(self, train_series, test_series, node_id, metric, lookback=3, epochs=100, batch_size=32):
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
        test_scaled = scaler.transform(test_series.values.reshape(-1, 1))
        self.scalers[f"{node_id}_{metric}"] = scaler
        
        X_train, y_train = self.create_sequences(train_scaled, lookback)
        X_test, y_test = self.create_sequences(test_scaled, lookback)
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = self.build_cnn_lstm_model(lookback)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                 callbacks=[early_stopping], verbose=0)
        
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        
        actual = test_series.values[lookback:]
        mape = self.calculate_mape(actual, predictions)
        
        return predictions.flatten(), mape