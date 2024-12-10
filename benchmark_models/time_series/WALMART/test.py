import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow to use dynamic memory allocation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

def load_and_prepare_data():
    try:
        stores = pd.read_csv('stores.csv')
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        
        train['Date'] = pd.to_datetime(train['Date'])
        test['Date'] = pd.to_datetime(test['Date'])
        
        train = train.merge(stores, on='Store', how='left')
        test = test.merge(stores, on='Store', how='left')
        
        daily_sales = train.groupby('Date')['Weekly_Sales'].sum().reset_index()
        daily_sales = daily_sales.sort_values('Date')
        
        return daily_sales['Weekly_Sales'].values, daily_sales['Date'].values
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        raise

def prepare_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out])
    return np.array(X), np.array(y)

class BasicSARIMAX:
    def __init__(self):
        self.model = None
        
    def fit(self, data):
        try:
            self.model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
            self.model = self.model.fit(disp=False)
        except Exception as e:
            print(f"Error fitting SARIMAX: {str(e)}")
            raise
        
    def predict(self, n_steps):
        try:
            return self.model.forecast(n_steps)
        except Exception as e:
            print(f"Error in SARIMAX prediction: {str(e)}")
            raise

class DeepLearningModels:
    def __init__(self, model_type='LSTM'):
        self.model = None
        self.model_type = model_type
        self.scaler = MinMaxScaler()
        
    def build_model(self, input_shape):
        try:
            model = tf.keras.Sequential()
            
            if self.model_type == 'LSTM':
                model.add(tf.keras.layers.LSTM(50, input_shape=input_shape, return_sequences=False))
            elif self.model_type == 'GRU':
                model.add(tf.keras.layers.GRU(50, input_shape=input_shape, return_sequences=False))
            elif self.model_type == 'CNN-LSTM':
                model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
                model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
                model.add(tf.keras.layers.LSTM(50))
            elif self.model_type == 'CNN-GRU':
                model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
                model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
                model.add(tf.keras.layers.GRU(50))
            
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model
        except Exception as e:
            print(f"Error building {self.model_type} model: {str(e)}")
            raise
    
    def fit(self, X, y):
        try:
            self.model = self.build_model((X.shape[1], 1))
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            self.model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
        except Exception as e:
            print(f"Error fitting {self.model_type}: {str(e)}")
            raise
        
    def predict(self, X):
        try:
            X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
            return self.model.predict(X_reshaped)
        except Exception as e:
            print(f"Error in {self.model_type} prediction: {str(e)}")
            raise

def evaluate_models(sales_data, dates):
    try:
        # Split data into train and test
        train_size = int(len(sales_data) * 0.8)
        train_sales = sales_data[:train_size]
        test_sales = sales_data[train_size:]
        
        # Parameters for sequence preparation
        n_steps_in = 12
        n_steps_out = 4
        
        # Prepare data for deep learning models
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(sales_data.reshape(-1, 1)).flatten()
        X, y = prepare_sequences(scaled_data, n_steps_in, n_steps_out)
        
        # Split sequences
        train_X = X[:train_size-n_steps_in]
        train_y = y[:train_size-n_steps_in].reshape(-1, 1)
        test_X = X[train_size-n_steps_in:]
        test_y = y[train_size-n_steps_in:].reshape(-1, 1)
        
        models = {
            'SARIMAX': BasicSARIMAX(),
            'LSTM': DeepLearningModels('LSTM'),
            'GRU': DeepLearningModels('GRU'),
            'CNN-LSTM': DeepLearningModels('CNN-LSTM'),
            'CNN-GRU': DeepLearningModels('CNN-GRU')
        }
        
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                if name in ['LSTM', 'GRU', 'CNN-LSTM', 'CNN-GRU']:
                    model.fit(train_X, train_y)
                    predictions = model.predict(test_X)
                    predictions = scaler.inverse_transform(predictions).flatten()
                    actual = scaler.inverse_transform(test_y).flatten()
                else:  # SARIMAX
                    model.fit(train_sales)
                    predictions = model.predict(len(test_sales))
                    actual = test_sales
                    
                # Calculate metrics
                mse = mean_squared_error(actual, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual, predictions)
                r2 = r2_score(actual, predictions)
                
                results[name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'predictions': predictions,
                    'actual': actual
                }
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                continue
            
        return results
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise

def plot_best_model(results):
    try:
        if not results:
            print("No models were successfully trained.")
            return
            
        # Find best model based on RMSE
        best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
        
        plt.figure(figsize=(15, 6))
        plt.plot(results[best_model]['actual'], label='Actual', color='blue')
        plt.plot(results[best_model]['predictions'], label='Predicted', color='red')
        plt.title(f'Best Model: {best_model} - Forecast vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Weekly Sales')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print performance report
        print("\nPerformance Report:")
        for model, metrics in results.items():
            print(f"\n{model}:")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"R2: {metrics['R2']:.2f}")
    except Exception as e:
        print(f"Error in plotting: {str(e)}")

if __name__ == "__main__":
    try:
        # Load and prepare data
        sales_data, dates = load_and_prepare_data()
        
        # Evaluate models
        results = evaluate_models(sales_data, dates)
        
        # Plot results and show performance report
        plot_best_model(results)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")