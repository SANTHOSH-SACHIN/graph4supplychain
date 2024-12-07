# Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)

Year: 2019  
Publication: AAAI  
Paper Link: https://ojs.aaai.org/index.php/AAAI/article/view/3881

## Model Characteristics

#### Task:

- Multistep (M-S)

#### Architecture:

- Discrete (D)
- Continuous (C)

#### Spatial Module:

- Spatial GNN

#### Temporal Module:

- Time Hybrid (T-H)

#### Missing Values:

- No

#### Input Graph:

- Required (R)

#### Learned Graph Relations:

- Not applicable

#### Graph Heuristics:

- Spatial Proximity (SP), Personal Computer (PC), Personal System (PS)

## Model Architecture

### Spatial Module

- **Graph Convolutional Networks (GCN)**: Used to capture the spatial dependencies among nodes in the traffic network.

### Attention Mechanisms(Temporal)

- **Spatial Attention (SAtt)**: Highlights important spatial features that are crucial for predicting traffic flow.
- **Temporal Attention (TAtt)**: Identifies significant temporal patterns, helping the model to focus on critical moments in historical data.

## Framework Overview

The framework of ASTGCN consists of multiple components designed to handle different aspects of the data:

- **ST blocks**: Each spatial-temporal block contains layers for both graph convolution and attention mechanisms, to process spatial and temporal data concurrently.
- **Fusion Layer**: Integrates features processed from different time components—recent, daily, and weekly patterns—to form a comprehensive feature set for prediction.
- **Loss Function**: The model optimizes a loss function that measures the accuracy of the traffic forecasts against actual observed traffic flows.

## Temporal Segmentation Strategy

To capture various periodic dependencies in the traffic data effectively, ASTGCN segments the input data into different temporal parts:

- **Recent Segment (X^(T))**: Captures the immediate past information, crucial for short-term forecasting.
- **Daily Segment (X^(D))**: Models daily periodic patterns, useful for understanding daily traffic cycles.
- **Weekly Segment (X^(W))**: Accounts for weekly periodic trends, important for capturing behaviors like weekend vs. weekday differences.

## Performance Metrics

| Model     | PeMSD4 RMSE | PeMSD4 MAE | PeMSD8 RMSE | PeMSD8 MAE |
| --------- | ----------- | ---------- | ----------- | ---------- |
| HA        | 54.14       | 36.76      | 44.03       | 29.52      |
| ARIMA     | 68.13       | 32.11      | 43.30       | 24.04      |
| VAR       | 51.73       | 33.76      | 31.21       | 21.41      |
| LSTM      | 45.82       | 29.45      | 36.96       | 23.18      |
| GRU       | 45.11       | 28.65      | 35.95       | 22.20      |
| STGCN     | 38.29       | 25.15      | 27.87       | 18.88      |
| GLU-STGCN | 38.41       | 27.28      | 30.78       | 20.99      |
| GeoMAN    | 37.84       | 23.64      | 28.91       | 17.84      |
| MSTGCN    | 35.64       | 22.73      | 26.47       | 17.47      |
| ASTGCN    | **32.82**   | **21.80**  | **25.27**   | **16.63**  |

## Hyperparameters

- **Chebyshev Polynomial (K)**: It was observed that as \( K \) increases, the forecasting performance slightly improves. The optimal value for \( K \) was set to 3 due to its balance between computational efficiency and performance improvement.
- **Kernel Size**: The kernel size in the temporal dimension is also set to 3 to optimize performance.
- **Convolution Kernels**: Each graph convolution layer utilizes 64 convolution kernels to process spatial features. Similarly, each temporal convolution layer also uses 64 convolution kernels.

- **Loss Function**: Mean Square Error (MSE) is used as the loss function, which measures the accuracy of predictions against the ground truth.
- **Optimization**: The loss is minimized through backpropagation during training.
- **Batch Size**: 64
- **Learning Rate**: Set at 0.0001 for optimal convergence.
