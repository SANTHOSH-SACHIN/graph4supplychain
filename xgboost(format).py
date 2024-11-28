# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

class SupplyChainForecaster:
    def __init__(self, metadata_path, timestamps_dir):
        self.metadata = self.load_metadata(metadata_path)
        self.timestamps_dir = timestamps_dir
        self.product_offerings = self.collect_product_offerings()

    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def collect_product_offerings(self):
        product_offerings = {}
        base_date = pd.to_datetime(self.metadata['base_date'])
        for i in range(1, self.metadata['total_timestamps']):
            timestamp_file = os.path.join(self.timestamps_dir, f'{i}.json')
            with open(timestamp_file, 'r') as f:
                timestamp_data = json.load(f)
            date = base_date + relativedelta(months=i)
            for offering in timestamp_data['node_values'].get('PRODUCT_OFFERING', []):
                po_id = offering[4]
                if po_id not in product_offerings:
                    product_offerings[po_id] = {'dates': [], 'costs': [], 'demands': []}
                product_offerings[po_id]['dates'].append(date)
                product_offerings[po_id]['costs'].append(offering[2])
                product_offerings[po_id]['demands'].append(offering[3])
        return product_offerings

    def prepare_features(self, dates, costs, timesteps=1):
        dates = pd.to_datetime(dates)
        features = pd.DataFrame({
            'cost': costs,
            'month': dates.month,
            'year': dates.year,
            'quarter': dates.quarter
        }, index=dates)
        for lag in range(1, timesteps + 1):
            features[f'cost_lag_{lag}'] = features['cost'].shift(lag)
        for window in [2, 3, 6]:
            features[f'cost_rolling_mean_{window}'] = features['cost'].rolling(window=window).mean()
        features['month_sin'] = np.sin(2 * np.pi * features.index.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * features.index.month / 12)
        y = features['cost'].shift(-timesteps)
        features = features.dropna()
        y = y.loc[features.index]
        features = features.iloc[:-timesteps]
        y = y.iloc[:-timesteps]
        print(features)
        return features, y.values

    def train_model(self, X_train, y_train):
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def forecast(self, model, X_test, future_features):
        test_predictions = model.predict(X_test)
        future_predictions = model.predict(future_features)
        return test_predictions, future_predictions

    def calculate_mape(self, y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred) * 100

    def plot_forecast_comparison(self, train_values, test_values, forecast, title, variable, node_id, mape):
        plt.figure(figsize=(12, 6))
        plt.plot(train_values.index, train_values, label='Train')
        plt.plot(test_values.index, test_values, label='Test')
        plt.plot(test_values.index, forecast, label='Forecast')
        plt.title(f'{title} - MAPE: {mape:.2f}%')
        plt.xlabel('Date')
        plt.ylabel(f'{variable} for Node {node_id}')
        plt.legend()
        plt.show()

    def prepare_future_features(self, last_date, last_cost, forecast_periods, timesteps):
        future_dates = [last_date + relativedelta(months=i + 1) for i in range(forecast_periods)]
        future_costs = [last_cost] * forecast_periods
        future_features, _ = self.prepare_features(future_dates, future_costs, timesteps)
        future_features.ffill(inplace=True)  # Use ffill instead of fillna(method='ffill')
        return future_features

def main():
    metadata_path = 'metadata.json'
    timestamps_dir = './'
    forecaster = SupplyChainForecaster(metadata_path, timestamps_dir)
    mape_results = pd.DataFrame(index=[1, 3, 6], columns=forecaster.product_offerings.keys())

    for timesteps in [1, 3, 6]:
        for product_id in forecaster.product_offerings.keys():
            try:
                dates = forecaster.product_offerings[product_id]['dates']
                costs = forecaster.product_offerings[product_id]['costs']
                features, y = forecaster.prepare_features(dates, costs, timesteps)
                if len(features) < 1 or len(y) < 1:
                    continue
                train_size = int(len(features) * 0.8)
                X_train = features.iloc[:train_size]
                X_test = features.iloc[train_size:]
                y_train = y[:train_size]
                y_test = y[train_size:]
                model = forecaster.train_model(X_train, y_train)
                future_features = forecaster.prepare_future_features(
                    pd.to_datetime(forecaster.product_offerings[product_id]['dates'][-1]),
                    forecaster.product_offerings[product_id]['costs'][-1],
                    forecast_periods=3,
                    timesteps=timesteps
                )
                test_predictions, future_predictions = forecaster.forecast(model, X_test, future_features)
                if len(y_test) > 0:
                    mape = forecaster.calculate_mape(y_test, test_predictions)
                    if isinstance(mape, float):
                        mape_results.loc[timesteps, product_id] = mape
                    else:
                        print(f"Invalid MAPE value for product {product_id} with timesteps {timesteps}")
                else:
                    print(f"Skipping product {product_id} due to insufficient test data for timesteps {timesteps}")
                train_values = pd.Series(y_train, index=X_train.index)
                test_values = pd.Series(y_test, index=X_test.index)
                forecaster.plot_forecast_comparison(
                    train_values,
                    test_values,
                    test_predictions,
                    f'Demand Forecast for Product {product_id}',
                    'Demand',
                    product_id,
                    mape
                )
            except Exception as e:
                print(f"Error forecasting for product {product_id} with timesteps {timesteps}: {e}")

    print("\nMAPE Summary Statistics:")
    for timesteps in [1, 3, 6]:
        print(f"\nTimesteps: {timesteps}")
        mapes = mape_results.loc[timesteps].dropna()
        print(f"Mean MAPE: {mapes.mean():.2f}%")
        print(f"Median MAPE: {mapes.median():.2f}%")
        print(f"Min MAPE: {mapes.min():.2f}%")
        print(f"Max MAPE: {mapes.max():.2f}%")

if __name__ == "__main__":
    main()