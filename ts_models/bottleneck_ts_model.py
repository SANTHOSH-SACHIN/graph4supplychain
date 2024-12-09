import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class POForecast:
    def __init__(self, df):
        self.df = df
    def function(self, forecast_steps):
        forecasts = {}
        for col in self.df.columns:
            series = self.df[col]
            
            try:
                model = ARIMA(series, order=(5, 1, 0))  # Example: ARIMA(5, 1, 0)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=forecast_steps)
                forecasts[col] = forecast.tolist()

            except Exception as e:
                forecasts[col] = f"Error: {e}"

        return forecasts
