# Time Series Forecasting

Graph4SupplyChain provides multiple time series forecasting models to predict supply chain metrics. This guide covers the available models, their use cases, and configuration options.

## Available Models

### 1. ARIMA (AutoRegressive Integrated Moving Average)
```python
from ts_models.arima_st import MultiStepARIMA

model = MultiStepARIMA(
    p=1,  # AR order
    d=1,  # Differencing order
    q=1   # MA order
)
```

- **Best for**:
  - Short to medium-term forecasting
  - Data with clear trends
  - Stationary or differenced series
- **Key Parameters**:
  - `p`: Autoregressive order
  - `d`: Differencing order
  - `q`: Moving average order
  - `seasonal`: Boolean for seasonal components

### 2. Prophet
```python
from ts_models.prophet_st import ProphetModel

model = ProphetModel(
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative'
)
```

- **Best for**:
  - Strong seasonal patterns
  - Missing data handling
  - Multiple seasonality
  - Holiday effects
- **Key Parameters**:
  - `changepoint_prior_scale`: Flexibility of trend
  - `seasonality_mode`: Additive or multiplicative
  - `holidays`: Custom holiday definitions
  - `interval_width`: Prediction interval width

### 3. XGBoost
```python
from ts_models.xgboost_st import XGBoostModel

model = XGBoostModel(
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1
)
```

- **Best for**:
  - Complex patterns
  - Multiple feature inputs
  - Non-linear relationships
- **Key Parameters**:
  - `max_depth`: Tree depth
  - `n_estimators`: Number of trees
  - `learning_rate`: Boosting learning rate
  - `early_stopping_rounds`: Prevent overfitting

### 4. SARIMA (Seasonal ARIMA)
```python
from ts_models.sarima_updated import SARIMAModel

model = SARIMAModel(
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)
```

- **Best for**:
  - Strong seasonal components
  - Regular patterns
  - Long-term forecasting
- **Key Parameters**:
  - `order`: (p,d,q) parameters
  - `seasonal_order`: (P,D,Q,s) seasonal parameters
  - `enforce_stationarity`: Boolean
  - `enforce_invertibility`: Boolean

## Model Selection Guide

Choose your model based on these factors:

1. **Data Characteristics**
   - Seasonality → Prophet or SARIMA
   - Multiple features → XGBoost
   - Simple trends → ARIMA
   - Complex patterns → XGBoost or Prophet

2. **Forecast Horizon**
   - Short-term → ARIMA
   - Medium-term → Prophet
   - Long-term → SARIMA or Prophet

3. **Data Quality**
   - Missing data → Prophet
   - Noisy data → XGBoost
   - Clean data → Any model

## Usage Examples

### Single Node Forecast
```python
# Load data
data = load_node_data(node_id)

# Initialize model
model = ProphetModel()

# Train and forecast
forecast = model.fit_predict(
    data,
    forecast_horizon=30
)
```

### Multi-Node Forecast
```python
# Load multiple nodes
nodes_data = load_multiple_nodes(node_ids)

# Parallel forecasting
forecasts = parallel_forecast(
    nodes_data,
    model_class=XGBoostModel,
    horizon=30
)
```

## Performance Metrics

Available metrics for model evaluation:

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: R-squared score

```python
from utils.metrics import calculate_metrics

metrics = calculate_metrics(
    actual_values,
    predicted_values
)
```

## Best Practices

1. **Data Preparation**
   - Remove outliers
   - Handle missing values
   - Normalize if needed
   - Split into train/test

2. **Model Validation**
   - Use cross-validation
   - Check residuals
   - Compare multiple models
   - Monitor performance

3. **Parameter Tuning**
   - Start with defaults
   - Use grid search
   - Monitor overfitting
   - Validate results

4. **Production Deployment**
   - Regular retraining
   - Performance monitoring
   - Error handling
   - Logging
