# SST-GNN: Simplified Spatio-Temporal Traffic Forecasting Model Using Graph Neural Network

Year: 2021  
Publication: PAKDD Conference  
Paper Link: https://link.springer.com/chapter/10.1007/978-3-030-75768-7_8

#### Task:

- Traffic Forecasting

#### Architecture:

- Simplified Spatio-Temporal Graph Neural Network (SST-GNN)

#### Spatial Module:

- Graph Neural Network (GNN)

#### Temporal Module:

- Weighted Spatio-Temporal Aggregation

#### Missing Values:

- No

#### Input Graph:

- Required

#### Learned Graph Relations:

- Static and Dynamic

#### Graph Heuristics:

- Multiple-hop Neighbors, Temporal Aggregation

# SST-GNN: Simplified Spatio-Temporal Traffic Forecasting Model Using Graph Neural Network

## 1. Introduction

The **SST-GNN** model captures spatial relationships and temporal dynamics in traffic data, focusing on road junctions at different hop-distances to encode distinct traffic information. This model aims to address limitations in traditional multi-layer GNNs and recurrent networks that struggle to capture long-range dependencies and periodic traffic patterns.

## 2. Methodology

The model is designed to:

- **Spatial Dependency**: Encoded using GNNs by separately aggregating different neighborhood representations rather than stacking multiple layers.
- **Temporal Dependency**: Captured using a simple yet effective spatio-temporal aggregation mechanism that models long-term dependencies.
- **Traffic Periodicity**: Captures daily and weekly traffic patterns using a novel position encoding scheme.

## 3. Model Components

The **SST-GNN** model includes:

- **Spatial Aggregation**: Handles different hop neighbors, focusing on their distinct impacts on traffic.
- **Temporal Aggregation**: Encodes the time-based influence by weighting the contribution of previous timestamps.
- **Historical and Current-Day Models**: Two models that separately capture daily and historical traffic patterns.

## 4. Hyperparameters

- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Time Window**: 60 minutes (12 timestamps)
- **Evaluation Metrics**: MAE, RMSE, MAPE

## 5. Performance Metrics

| Dataset | 15-min Forecast | 30-min Forecast | 45-min Forecast | 60-min Forecast |
| ------- | --------------- | --------------- | --------------- | --------------- |
| PeMSD7  | MAE: 2.04       | MAE: 2.67       | MAE: 3.17       | MAE: 3.48       |
| PeMSD4  | MAE: 1.23       | MAE: 1.82       | MAE: 1.84       | MAE: 2.13       |
| PeMSD8  | MAE: 1.03       | MAE: 1.39       | MAE: 1.62       | MAE: 1.74       |

These datasets contain traffic speed information from different sensors on road networks, recorded at 5-minute intervals.
