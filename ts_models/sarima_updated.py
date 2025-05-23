import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_percentage_error, 
    mean_absolute_error, 
    mean_squared_error
)
from typing import Dict, List, Tuple
import streamlit as st 

def sarima_demand_forecast(train_demand, test_demand, 
                                         order=(2,1,2), 
                                         seasonal_order=(2,1,2,6), 
                                         verbose=True):
    """
    Perform single-step demand forecasting using SARIMA model with enhanced visualization.

    Parameters:
    -----------
    train_demand : pandas.Series
        Training time series data with datetime index for the selected part.
    test_demand : pandas.Series
        Testing time series data with datetime index for the selected part.
    order : tuple, optional
        SARIMA non-seasonal order (p,d,q).
    seasonal_order : tuple, optional
        SARIMA seasonal order (P,D,Q,s).
    verbose : bool, optional
        Print additional diagnostic information.

    Returns:
    --------
    dict: A dictionary containing:
    - forecasts: Forecasted values.
    - actual: Actual test values.
    - mape: Mean Absolute Percentage Error.
    - mae: Mean Absolute Error.
    - rmse: Root Mean Squared Error.
    - plot: Matplotlib figure with subplots for detailed visualization.
    """
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    import matplotlib.pyplot as plt

    if not isinstance(train_demand.index, pd.DatetimeIndex):
        train_demand.index = pd.to_datetime(train_demand.index)
    if not isinstance(test_demand.index, pd.DatetimeIndex):
        test_demand.index = pd.to_datetime(test_demand.index)
    
    if len(train_demand) == 0 or len(test_demand) == 0:
        raise ValueError("Train or test data is empty.")
    
    try:
        model = SARIMAX(train_demand, 
                        order=order, 
                        seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        st.write(f"Model AIC: {model_fit.aic}")
        st.write(f"Model BIC: {model_fit.bic}")
        if verbose:
            print(model_fit.summary())
        
        forecast_steps = len(test_demand)
        forecast = model_fit.get_forecast(steps=forecast_steps)

        forecast_mean = forecast.predicted_mean
        
        residuals = train_demand - model_fit.fittedvalues
        
        mape = mean_absolute_percentage_error(test_demand, forecast_mean)
        mae = mean_absolute_error(test_demand, forecast_mean)
        rmse = np.sqrt(mean_squared_error(test_demand, forecast_mean))
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        
        axs[0].plot(train_demand.index, train_demand, 
                    label='Training Data', color='blue', alpha=0.7)
        axs[0].plot(test_demand.index, test_demand, 
                    label='Actual Test Data', color='green', alpha=0.7)
        axs[0].set_title('Training and Actual Data')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Demand')
        axs[0].legend()
        
        axs[1].plot(test_demand.index, test_demand, 
                    label='Actual Test Data', color='green', alpha=0.7)
        axs[1].plot(test_demand.index, forecast_mean, 
                    color='red', label='Forecast', linestyle='--')
    
        conf_int = forecast.conf_int()
        axs[1].fill_between(test_demand.index, 
                            conf_int.iloc[:, 0], 
                            conf_int.iloc[:, 1], 
                            color='pink', alpha=0.3)
        axs[1].set_title('Forecast vs Actual with Confidence Interval')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Demand')
        axs[1].legend()
        
        axs[2].plot(residuals.index, residuals, 
                    label='Residuals', color='purple', alpha=0.7)
        axs[2].axhline(0, linestyle='--', color='black', alpha=0.5)
        axs[2].set_title('Residuals')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Residuals')
        axs[2].legend()
        
        plt.tight_layout()
        
        plot = plt.gcf()
        plt.close()
        
        results = {
            'forecasts': forecast_mean,
            'actual': test_demand,
            'mape': mape * 100,  
            'mae': mae,
            'rmse': rmse,
            'plot': plot,
            'model_summary': model_fit.summary() if verbose else None
        }
        return results['forecasts'],results['mape']
    
    except Exception as e:
        print(f"Error during forecasting: {e}")
        return None

    
def forecast_node_multistep_sarima(train_series: pd.Series, test_series: pd.Series,
                                    forecast_horizons: List[int], 
                                    order: Tuple[int, int, int]=(1, 1, 1),
                                    seasonal_order: Tuple[int, int, int, int]=(1, 1, 1, 7)) -> Tuple[List[np.ndarray], Dict[int, float]]:
    """
    Fit SARIMA model and generate multi-step forecasts for different horizons
    
    Parameters:
    - train_series: Training time series data
    - test_series: Testing time series data
    - forecast_horizons: List of forecast steps to predict
    - order: SARIMA non-seasonal order (p,d,q)
    - seasonal_order: SARIMA seasonal order (P,D,Q,s)
    
    Returns:
    - forecasts: List of forecasted values for each horizon
    - mapes: Dictionary of MAPE values for each forecast horizon
    """
    
    if not isinstance(train_series.index, pd.DatetimeIndex):
        train_series.index = pd.to_datetime(train_series.index)
    if not isinstance(test_series.index, pd.DatetimeIndex):
        test_series.index = pd.to_datetime(test_series.index)
    
    full_series = pd.concat([train_series, test_series])
    
    def calculate_mape(actual, forecast):
        """
        Calculate Mean Absolute Percentage Error
        
        Parameters:
        - actual: Actual values
        - forecast: Forecasted values
        
        Returns:
        - MAPE value
        """
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100

    forecasts = []
    mapes = {}
    
    for horizon in forecast_horizons:
        horizon_forecast = []
        
        for i in range(len(test_series) - horizon + 1):
          
            train_data = full_series.iloc[:len(train_series) + i]
            
            try:
                model = SARIMAX(train_data, 
                                order=order, 
                                seasonal_order=seasonal_order)
                results = model.fit(disp=False)

                forecast = results.get_forecast(steps=horizon)
                forecast_mean = forecast.predicted_mean
                
                horizon_forecast.append(forecast_mean.iloc[-1])
            
            except Exception as e:
                print(f"Error in forecasting for horizon {horizon}: {e}")
                horizon_forecast.append(train_data.iloc[-1])
       
        horizon_forecast = np.array(horizon_forecast)
        forecasts.append(horizon_forecast)
        
        actual = test_series.iloc[horizon-1:].values
        mapes[horizon] = calculate_mape(actual, horizon_forecast)
    
    return forecasts, mapes