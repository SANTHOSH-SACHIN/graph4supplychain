# Time Series Models API Reference

This document provides detailed API documentation for the time series forecasting models in Graph4SupplyChain.

## MultiStepARIMA

```python
from ts_models.arima_st import MultiStepARIMA
```

Class for multi-step ARIMA forecasting with configurable parameters.

### Parameters

- `p` (int, default=1): The order of the AR term
- `d` (int, default=1): The order of differencing
- `q` (int, default=1): The order of the MA term
- `seasonal` (bool, default=False): Whether to include seasonal components

### Methods

#### fit(data)
Fits the ARIMA model to the training data.

```python
def fit(self, data: pd.DataFrame) -> None:
    """
    Fit the ARIMA model.
    
    Parameters:
        data (pd.DataFrame): Training data with datetime index
    """
```

#### forecast_node_multistep(data, steps)
Generate multi-step forecasts for a node.

```python
def forecast_node_multistep(
    self,
    data: pd.DataFrame,
    steps: int
) -> pd.DataFrame:
    """
    Generate multi-step forecasts.
    
    Parameters:
        data (pd.DataFrame): Historical data
        steps (int): Number of steps to forecast
        
    Returns:
        pd.DataFrame: Forecasted values with confidence intervals
    """
```

## ProphetModel

```python
from ts_models.prophet_st import ProphetModel
```

Facebook Prophet model implementation for supply chain forecasting.

### Parameters

- `changepoint_prior_scale` (float, default=0.05): Flexibility of the trend
- `seasonality_mode` (str, default='multiplicative'): Type of seasonality
- `interval_width` (float, default=0.95): Width of prediction intervals

### Methods

#### fit(data)
Fits the Prophet model to the data.

```python
def fit(self, data: pd.DataFrame) -> None:
    """
    Fit the Prophet model.
    
    Parameters:
        data (pd.DataFrame): Training data with 'ds' and 'y' columns
    """
```

#### predict(periods)
Generate forecasts for specified periods.

```python
def predict(self, periods: int) -> pd.DataFrame:
    """
    Generate forecasts.
    
    Parameters:
        periods (int): Number of periods to forecast
        
    Returns:
        pd.DataFrame: Forecasted values with components
    """
```

## XGBoostModel

```python
from ts_models.xgboost_st import XGBoostModel
```

XGBoost-based time series forecasting model.

### Parameters

- `max_depth` (int, default=6): Maximum tree depth
- `n_estimators` (int, default=100): Number of boosting rounds
- `learning_rate` (float, default=0.1): Boosting learning rate
- `objective` (str, default='reg:squarederror'): Learning objective

### Methods

#### fit(X, y)
Fits the XGBoost model to the training data.

```python
def fit(
    self,
    X: np.ndarray,
    y: np.ndarray
) -> None:
    """
    Fit the XGBoost model.
    
    Parameters:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
    """
```

#### predict(X)
Generate predictions for new data.

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions.
    
    Parameters:
        X (np.ndarray): Feature matrix
        
    Returns:
        np.ndarray: Predicted values
    """
```

## SARIMAModel

```python
from ts_models.sarima_updated import SARIMAModel
```

Seasonal ARIMA model for time series with regular patterns.

### Parameters

- `order` (tuple): (p,d,q) parameters
- `seasonal_order` (tuple): (P,D,Q,s) seasonal parameters
- `enforce_stationarity` (bool, default=True): Whether to enforce stationarity
- `enforce_invertibility` (bool, default=True): Whether to enforce invertibility

### Methods

#### fit(data)
Fits the SARIMA model to the data.

```python
def fit(self, data: pd.DataFrame) -> None:
    """
    Fit the SARIMA model.
    
    Parameters:
        data (pd.DataFrame): Training data with datetime index
    """
```

#### forecast(steps)
Generate forecasts for specified steps.

```python
def forecast(self, steps: int) -> pd.DataFrame:
    """
    Generate forecasts.
    
    Parameters:
        steps (int): Number of steps to forecast
        
    Returns:
        pd.DataFrame: Forecasted values with confidence intervals
    """
```

## Utility Functions

### calculate_metrics
```python
from utils.metrics import calculate_metrics

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> dict:
    """
    Calculate performance metrics.
    
    Parameters:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary of metrics (MAE, RMSE, MAPE, R²)
    """
```

### prepare_data
```python
from utils.data_preparation import prepare_data

def prepare_data(
    data: pd.DataFrame,
    target_col: str,
    sequence_length: int
) -> tuple:
    """
    Prepare data for time series modeling.
    
    Parameters:
        data (pd.DataFrame): Raw data
        target_col (str): Target column name
        sequence_length (int): Length of input sequences
        
    Returns:
        tuple: (X, y) prepared features and targets
    """
```
