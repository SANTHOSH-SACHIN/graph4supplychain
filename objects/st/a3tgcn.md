# A3TGCN

Year: 2021  
Publication: ISPRS  
Paper Link: https://www.mdpi.com/2220-9964/10/7/485

#### Task:

- Multistep (M-S)

#### Architecture:

- Discrete (D)
- Factorised (F)

#### Spatial Module:

- Spatial GNNc

#### Temporal Module:

- Temporal GRU (T-G)

#### Missing Values:

- No

#### Input Graph:

- Required (R)

#### Learned Graph Relations:

- Static (S)

#### Graph Heuristics:

- Spatial Proximity (SP), Pairwise Connectivity (PC)

# A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting

## 1. Methodology

The **A3T-GCN** model integrates Graph Convolutional Networks (GCNs) and Gated Recurrent Units (GRUs) with an attention mechanism to effectively model both spatial and temporal dependencies in traffic data. The model works as follows:

- **Spatial Dependencies**: Captured using GCN layers that process the topological structure of road networks, identifying traffic patterns based on connectivity and proximity between different locations.
- **Temporal Dependencies**: GRU layers are employed to model the sequential nature of traffic data, taking into account past time steps and learning temporal dynamics.
- **Attention Mechanism**: The attention layer helps the model focus on the most relevant historical time points, assigning weights to enhance the predictive accuracy by emphasizing more influential time steps.

## 2. Layers Used

The model architecture of **A3T-GCN** consists of the following types and numbers of layers:

- **Graph Convolutional Layers (GCNs)**: These layers are used to capture the spatial relationships between nodes (e.g., different traffic points on the road network).
- **Gated Recurrent Unit (GRU) Layers**: The GRU layers focus on capturing the temporal dynamics, with recurrent connections allowing the model to remember past traffic conditions and make accurate predictions for future time steps.
- **Attention Layer**: A single attention layer is applied to learn the importance of each past time step, enhancing the model’s ability to focus on relevant data.

The exact number of layers and configuration is as follows:

- **2 GCN Layers** for spatial feature extraction.
- **2 GRU Layers** for temporal processing.
- **1 Attention Layer** to weigh the importance of the input data dynamically.

## 3. Model Architecture

The **A3T-GCN** model combines three major components:

- **Graph Convolutional Networks (GCNs)**: Used for learning spatial dependencies in traffic networks, where each node represents a location (such as a sensor or intersection) and the edges represent the connections (e.g., roads) between them.
- **Gated Recurrent Units (GRUs)**: These units learn the temporal dependencies by considering traffic conditions at previous time steps, adjusting their predictions based on sequential patterns.
- **Attention Mechanism**: This mechanism assigns weights to the output of the GRU layers, focusing on more relevant historical data points.

The complete model architecture follows this flow:

1. **GCN Layer 1** → **GCN Layer 2**: Extract spatial features from the road network.
2. **GRU Layer 1** → **GRU Layer 2**: Learn temporal dependencies from sequential traffic data.
3. **Attention Layer**: Apply weights to the output of the GRU layers, emphasizing important time steps.
4. **Fully Connected Layer**: Final prediction of traffic flow.

## 4. Hyperparameters

The following hyperparameters are used in the **A3T-GCN** model:

- **Learning Rate**: 0.001 (default)
- **Batch Size**: 64
- **Dropout Rate**: 0.3
- **Optimizer**: Adam optimizer is used to train the model.
- **GCN Filter Size**: 64
- **Number of GRU Units**: 128
- **Epochs**: 200 (with early stopping based on validation loss)

## 5. Performance Metrics

| Dataset  | Time   | Metric   | HA     | ARIMA   | SVR    | GCN    | GRU    | A3T-GCN    |
| -------- | ------ | -------- | ------ | ------- | ------ | ------ | ------ | ---------- |
| SZ-taxi  | 15 min | RMSE     | 4.2951 | 7.2406  | 4.1455 | 5.6596 | 3.9994 | **3.8989** |
|          |        | MAE      | 2.7815 | 4.9824  | 2.6233 | 4.2367 | 2.5955 | **2.6840** |
|          |        | Accuracy | 0.7008 | 0.4463  | 0.7112 | 0.6107 | 0.7249 | **0.7318** |
|          |        | R²       | 0.8307 | \*      | 0.8423 | 0.6654 | 0.8329 | **0.8512** |
|          |        | var      | 0.8307 | 0.0035  | 0.8424 | 0.6655 | 0.8329 | **0.8512** |
|          | 30 min | RMSE     | 4.2951 | 6.7899  | 4.1628 | 5.6918 | 4.0942 | **3.9228** |
|          |        | MAE      | 2.7815 | 4.6765  | 2.6875 | 4.2647 | 2.6906 | **2.7038** |
|          |        | Accuracy | 0.7008 | 0.3845  | 0.7100 | 0.6085 | 0.7184 | **0.7302** |
|          |        | R²       | 0.8307 | \*      | 0.8410 | 0.6616 | 0.8249 | **0.8493** |
|          |        | var      | 0.8307 | 0.0081  | 0.8413 | 0.6617 | 0.8250 | **0.8493** |
|          | 45 min | RMSE     | 4.2951 | 6.7852  | 4.1885 | 5.7142 | 4.1534 | **3.9461** |
|          |        | MAE      | 2.7815 | 4.6734  | 2.7359 | 4.2844 | 2.7743 | **2.7261** |
|          |        | Accuracy | 0.7008 | 0.3847  | 0.7082 | 0.6069 | 0.7143 | **0.7286** |
|          |        | R²       | 0.8307 | \*      | 0.8391 | 0.6589 | 0.8198 | **0.8474** |
|          |        | var      | 0.8307 | 0.0087  | 0.8397 | 0.6590 | 0.8199 | **0.8474** |
|          | 60 min | RMSE     | 4.2951 | 6.7708  | 4.2156 | 5.7361 | 4.0747 | **3.9707** |
|          |        | MAE      | 2.7815 | 4.6655  | 2.7751 | 4.3034 | 2.7712 | **2.7391** |
|          |        | Accuracy | 0.7008 | 0.3851  | 0.7063 | 0.6054 | 0.7197 | **0.7269** |
|          |        | R²       | 0.8307 | \*      | 0.8370 | 0.6564 | 0.8266 | **0.8454** |
|          |        | var      | 0.8307 | 0.0111  | 0.8379 | 0.6564 | 0.8267 | **0.8454** |
| Los-loop | 15 min | RMSE     | 7.4427 | 10.0439 | 6.0084 | 7.7922 | 5.2182 | **5.0904** |
|          |        | MAE      | 4.0145 | 7.6832  | 3.7285 | 5.3525 | 3.0602 | **3.1365** |
|          |        | Accuracy | 0.8733 | 0.8275  | 0.8977 | 0.8673 | 0.9109 | **0.9133** |
|          |        | R²       | 0.7121 | 0.0025  | 0.8123 | 0.6843 | 0.8576 | **0.8653** |
|          |        | var      | 0.7121 | \*      | 0.8146 | 0.6844 | 0.8577 | **0.8653** |
|          | 30 min | RMSE     | 7.4427 | 9.3450  | 6.9588 | 8.3353 | 6.2802 | **5.9974** |
|          |        | MAE      | 4.0145 | 7.6891  | 3.7248 | 5.6118 | 3.6505 | **3.6610** |
|          |        | Accuracy | 0.8733 | 0.8275  | 0.8815 | 0.8581 | 0.8931 | **0.8979** |
|          |        | R²       | 0.7121 | 0.0031  | 0.7492 | 0.6402 | 0.7957 | **0.8137** |
|          |        | var      | 0.7121 | \*      | 0.7523 | 0.6404 | 0.7958 | **0.8137** |
|          | 45 min | RMSE     | 7.4427 | 10.0508 | 7.7504 | 8.8036 | 7.0343 | **6.6840** |
|          |        | MAE      | 4.0145 | 7.6924  | 4.1288 | 5.9534 | 4.0915 | **4.1712** |
|          |        | Accuracy | 0.8733 | 0.8273  | 0.8680 | 0.8500 | 0.8801 | **0.8861** |
|          |        | R²       | 0.7121 | \*      | 0.6899 | 0.5999 | 0.7446 | **0.7694** |
|          |        | var      | 0.7121 | 0.0035  | 0.6947 | 0.6001 | 0.7451 | **0.7705** |
|          | 60 min | RMSE     | 7.4427 | 10.0538 | 8.4388 | 9.2657 | 7.6621 | **7.0990** |
|          |        | MAE      | 4.0145 | 7.6952  | 4.5036 | 6.2892 | 4.5186 | **4.2343** |
|          |        | Accuracy | 0.8733 | 0.8273  | 0.8562 | 0.8421 | 0.8694 | **0.8790** |
|          |        | R²       | 0.7121 | \*      | 0.6336 | 0.5583 | 0.6980 | **0.7407** |
|          |        | var      | 0.7121 | 0.0036  | 0.5593 | 0.5593 | 0.6984 | **0.7415** |

## Models Compared

In this study, the performance of AT-GCN is compared with several traditional and contemporary models across different metrics and time intervals. The models compared include:

- **HA (Historical Average)**
- **ARIMA (AutoRegressive Integrated Moving Average)**
- **SVR (Support Vector Regression)**
- **GCN (Graph Convolutional Network)**

## 6. Datasets Used

The **A3T-GCN** model is evaluated on two real-world traffic datasets:

1. **Shenzhen Dataset**:

   - **Location**: Shenzhen, China.
   - **Traffic Data**: Contains traffic speed data collected from various sensors across the city's road network.
   - **Time Period**: Several months of data, with traffic speeds recorded at 5-minute intervals.

2. **Los Angeles (PeMS) Dataset**:
   - **Location**: Los Angeles, California.
   - **Traffic Data**: Contains data from highway sensors that capture traffic flow and speed.
   - **Time Period**: Multiple months of data, aggregated over 5-minute intervals.

These datasets include both spatial data (the connectivity of roads) and temporal data (traffic patterns over time), making them suitable for evaluating models that rely on spatial-temporal features.
