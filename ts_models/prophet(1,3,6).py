import pandas as pd
import numpy as np
from prophet import Prophet
import json
import os
from sklearn.model_selection import train_test_split
from dateutil.relativedelta import relativedelta

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

class PRODUCT_OFFERINGForecast:
    def __init__(self, metadata_path, timestamps_dir):
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Prepare data structures
        self.timestamps = self.metadata['total_timestamps']
        self.timestamps_dir = timestamps_dir
        self.base_date = pd.to_datetime(self.metadata['base_date'])

        # Collect all product offerings across timestamps
        self.product_offerings = self.collect_product_offerings()

    def collect_product_offerings(self):
        product_offerings = {}

        for i in range(1, self.timestamps):
            timestamp_file = os.path.join(self.timestamps_dir, f'{i}.json')

            with open(timestamp_file, 'r') as f:
                timestamp_data = json.load(f)

            # Calculate the date for this timestamp
            date = self.base_date + relativedelta(months=i)

            # Process PRODUCT_OFFERING nodes
            for offering in timestamp_data['node_values'].get('PRODUCT_OFFERING', []):
                po_id = offering[4]  # ID is the last element
                name = offering[1]   # Name is the second element
                demand = offering[3]  # Demand is the fourth element

                if po_id not in product_offerings:
                    product_offerings[po_id] = {
                        'name': name,
                        'dates': [],
                        'demands': []
                    }

                product_offerings[po_id]['dates'].append(date)
                product_offerings[po_id]['demands'].append(demand)

        return product_offerings

    def prepare_prophet_dataframe(self, product_id):
        product = self.product_offerings[product_id]

        df = pd.DataFrame({
            'ds': pd.to_datetime(product['dates']),
            'y': product['demands']
        })
        print(df)
        return df.sort_values('ds')

    def add_external_regressors(self, df, product_id):
        features = []
        for i in range(1, self.timestamps):
            timestamp_file = os.path.join(self.timestamps_dir, f'{i}.json')

            with open(timestamp_file, 'r') as f:
                timestamp_data = json.load(f)

            for offering in timestamp_data['node_values'].get('PRODUCT_OFFERING', []):
                if offering[4] == product_id:
                    cost = offering[2]
                    features.append(cost)
                    break

        if features:
            df['cost'] = features

        return df

    def forecast_product_demand(self, product_id, forecast_periods=3, test_size=0.2):
        # Prepare base dataframe
        df = self.prepare_prophet_dataframe(product_id)

        # Add external regressors
        df = self.add_external_regressors(df, product_id)

        # Split data into train and test sets
        train_size = int(len(df) * (1 - test_size))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        # Initialize and fit Prophet model
        model = Prophet()

        if 'cost' in df.columns:
            model.add_regressor('cost')

        model.fit(train_df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=len(test_df) + forecast_periods)

        if 'cost' in df.columns:
            future['cost'] = np.linspace(
                train_df['cost'].mean(),
                train_df['cost'].max(),
                len(future)
            )

        # Make forecast
        forecast = model.predict(future)

        # Get predictions for test set
        test_predictions = forecast.iloc[train_size:train_size+len(test_df)]['yhat']

        # Calculate MAPE
        mape = mean_absolute_percentage_error(test_df['y'], test_predictions)

        return {
            'name': self.product_offerings[product_id]['name'],
            'forecast': forecast,
            'mape': mape,
            'test_actual': test_df['y'],
            'test_predictions': test_predictions
        }

    def forecast_all_products(self, forecast_periods=3, test_size=0.2):
        forecasts = {}

        for product_id in self.product_offerings.keys():
            try:
                forecast_result = self.forecast_product_demand(
                    product_id,
                    forecast_periods,
                    test_size
                )
                forecasts[product_id] = forecast_result
            except Exception as e:
                print(f"Could not forecast for product {product_id}: {e}")

        return forecasts

    def summarize_forecasts(self, forecasts):
        summary_data = []

        for product_id, result in forecasts.items():
            summary_data.append({
                'Product ID': product_id,
                'Product Name': result['name'],
                'MAPE (%)': result['mape'],
                'Avg Actual Demand': np.mean(result['test_actual']),
                'Avg Predicted Demand': np.mean(result['test_predictions'])
            })

        return pd.DataFrame(summary_data).sort_values('MAPE (%)')

def run_forecasts_with_different_steps():
    # Define folder path
    base_path = '/content/drive/MyDrive/NSS_1000_24'
    metadata_path = os.path.join(base_path, 'metadata.json')

    # Initialize forecasters for different step sizes
    forecaster_1step = PRODUCT_OFFERINGForecast(
        metadata_path=metadata_path,
        timestamps_dir=base_path
    )

    forecaster_3step = PRODUCT_OFFERINGForecast(
        metadata_path=metadata_path,
        timestamps_dir=base_path
    )

    forecaster_6step = PRODUCT_OFFERINGForecast(
        metadata_path=metadata_path,
        timestamps_dir=base_path
    )

    # Run forecasts with different step sizes
    print("Running 1-step forecast...")
    forecasts_1step = forecaster_1step.forecast_all_products(
        forecast_periods=1,
        test_size=0.2
    )
    summary_1step = forecaster_1step.summarize_forecasts(forecasts_1step)

    print("\nRunning 3-step forecast...")
    forecasts_3step = forecaster_3step.forecast_all_products(
        forecast_periods=3,
        test_size=0.2
    )
    summary_3step = forecaster_3step.summarize_forecasts(forecasts_3step)

    print("\nRunning 6-step forecast...")
    forecasts_6step = forecaster_6step.forecast_all_products(
        forecast_periods=6,
        test_size=0.2

    )
    summary_6step = forecaster_6step.summarize_forecasts(forecasts_6step)

    # Create output directory
    output_dir = os.path.join(base_path, 'forecast_results')

    os.makedirs(output_dir, exist_ok=True)

    # Save results function
    def save_forecast_results(forecasts, summary, step_size):
        summary_file = os.path.join(output_dir, f'summary_{step_size}step.csv')
        summary.to_csv(summary_file)

        for product_id, forecast_data in forecasts.items():
            forecast_file = os.path.join(output_dir, f'forecast_{step_size}step_product_{product_id}.csv')
            forecast_data['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file)

    # Save all results
    save_forecast_results(forecasts_1step, summary_1step, 1)
    save_forecast_results(forecasts_3step, summary_3step, 3)
    save_forecast_results(forecasts_6step, summary_6step, 6)

    return {
        '1step': {'forecasts': forecasts_1step, 'summary': summary_1step},
        '3step': {'forecasts': forecasts_3step, 'summary': summary_3step},
        '6step': {'forecasts': forecasts_6step, 'summary': summary_6step}
    }

if __name__ == "__main__":
    # Run all forecasts
    results = run_forecasts_with_different_steps()

    # Print comparative results
    print("\nComparative MAPE Summary:")
    print("\n1-Step Forecast Summary:")
    print(results['1step']['summary'][['Product Name', 'MAPE (%)']])

    print("\n3-Step Forecast Summary:")
    print(results['3step']['summary'][['Product Name', 'MAPE (%)']])

    print("\n6-Step Forecast Summary:")
    print(results['6step']['summary'][['Product Name', 'MAPE (%)']])

