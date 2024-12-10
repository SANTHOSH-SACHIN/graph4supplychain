# Basic Usage Examples

This guide provides practical examples of using Graph4SupplyChain for common supply chain analytics tasks.

## 1. Load and Visualize Supply Chain Data

### Load Data
```python
import pandas as pd
import networkx as nx
from utils.data_loader import load_supply_chain_data

# Load node and edge data
nodes_df = pd.read_csv('data/nodes.csv')
edges_df = pd.read_csv('data/edges.csv')

# Create graph
G = nx.from_pandas_edgelist(
    edges_df,
    source='source',
    target='target',
    edge_attr='weight'
)
```

### Visualize Network
```python
import streamlit as st
from utils.visualization import plot_network

# Plot interactive network
st.plotly_chart(
    plot_network(G, 
                 node_size='demand',
                 color_by='type')
)
```

## 2. Time Series Forecasting

### Single Node Forecast
```python
from ts_models.prophet_st import ProphetModel

# Prepare data for a single node
node_data = nodes_df[nodes_df['node_id'] == 'store_001']

# Initialize and train Prophet model
model = ProphetModel(
    changepoint_prior_scale=0.05,
    seasonality_mode='multiplicative'
)

# Generate forecast
forecast = model.fit_predict(
    node_data,
    forecast_horizon=30
)

# Plot results
st.plotly_chart(
    model.plot_forecast(forecast)
)
```

### Multi-Node Forecast
```python
from ts_models.xgboost_st import XGBoostModel

# Select multiple nodes
node_ids = ['store_001', 'store_002', 'warehouse_001']
multi_node_data = nodes_df[nodes_df['node_id'].isin(node_ids)]

# Initialize XGBoost model
model = XGBoostModel(
    max_depth=6,
    n_estimators=100
)

# Generate forecasts for all nodes
forecasts = {}
for node_id in node_ids:
    node_data = multi_node_data[multi_node_data['node_id'] == node_id]
    forecast = model.fit_predict(node_data)
    forecasts[node_id] = forecast
```

## 3. Network Analysis

### Calculate Centrality Metrics
```python
import networkx as nx

# Calculate various centrality metrics
centrality_metrics = {
    'degree': nx.degree_centrality(G),
    'betweenness': nx.betweenness_centrality(G),
    'closeness': nx.closeness_centrality(G)
}

# Add metrics to node attributes
for metric_name, metric_values in centrality_metrics.items():
    nx.set_node_attributes(G, metric_values, metric_name)
```

### Find Critical Paths
```python
from utils.network_analysis import find_critical_path

# Find critical path between supplier and store
critical_path = find_critical_path(
    G,
    source='supplier_001',
    target='store_001',
    weight='delivery_time'
)

# Visualize critical path
st.plotly_chart(
    plot_network(G, highlight_path=critical_path)
)
```

## 4. Performance Analysis

### Calculate Metrics
```python
from utils.metrics import calculate_metrics

# Calculate forecast accuracy
metrics = calculate_metrics(
    actual_values=node_data['demand'],
    predicted_values=forecast['yhat']
)

# Display metrics
st.write("Forecast Performance Metrics:")
st.write(f"MAE: {metrics['mae']:.2f}")
st.write(f"RMSE: {metrics['rmse']:.2f}")
st.write(f"MAPE: {metrics['mape']:.2f}%")
```

### Compare Models
```python
from ts_models.arima_st import MultiStepARIMA
from ts_models.prophet_st import ProphetModel
from utils.model_comparison import compare_models

# Initialize models
models = {
    'ARIMA': MultiStepARIMA(p=1, d=1, q=1),
    'Prophet': ProphetModel(),
    'XGBoost': XGBoostModel()
}

# Compare model performance
comparison = compare_models(
    models,
    data=node_data,
    target_col='demand',
    horizon=30
)

# Plot comparison
st.plotly_chart(
    comparison.plot_forecasts()
)
```

## 5. Export Results

### Save Forecasts
```python
# Export forecasts to CSV
for node_id, forecast in forecasts.items():
    forecast.to_csv(f'results/forecast_{node_id}.csv')

# Export network metrics
nx.write_gexf(G, 'results/network_metrics.gexf')
```

### Generate Report
```python
from utils.reporting import generate_report

# Generate PDF report
report = generate_report(
    forecasts=forecasts,
    metrics=metrics,
    network_stats=centrality_metrics
)

# Save report
report.save('results/supply_chain_analysis.pdf')
```

## Next Steps

- Explore [Advanced Features](advanced-features.md)
- Check the [API Reference](../api/time-series-models.md)
- Read about [Best Practices](../user-guide/overview.md)
