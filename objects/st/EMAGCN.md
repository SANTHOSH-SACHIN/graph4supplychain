
# Exponential Moving Average Graph Convolutional Networks (EMAGCN)

Year: 2023  
Publication: IEEE Transactions on Big Data  
Paper Link: [https://ieeexplore.ieee.org/document/10064498](https://ieeexplore.ieee.org/document/10064498)

## Model Characteristics

#### Task:

- Short-Term Load Forecasting (STLF)

#### Architecture:

- Discrete (D)
- Exponential Moving Average (EMA) based GCN

#### Spatial Module:

- Dynamic Graph Convolution (GNN)

#### Temporal Module:

- Temporal Embeddings (T-E)

#### Missing Values:

- No

#### Input Graph:

- Required (R)

#### Learned Graph Relations:

- Static and Dynamic Graphs

#### Graph Heuristics:

- Exponential Moving Average (EMA), Dynamic Graphs (DG)

## 1. Introduction

EMAGCN (Exponential Moving Average Graph Convolutional Network) is a core component of the STEGNN model, designed to mitigate the over-smoothing problem common in deep GCNs. The primary contributions of this model are:

- **Directed Dynamic Graphs**: Richer information compared to undirected graphs.
- **EMA-GCN**: Combines exponential moving average with GCN to improve prediction accuracy.
- **Trainable Temporal Embeddings**: Captures periodicity in load data effectively.

---

## 2. EMAGCN Structure

### Static and Dynamic Graph Construction
The model constructs both static and dynamic graphs, represented by adjacency matrices. The dynamic graph captures time-varying spatial dependencies, while the static graph captures long-term relationships between nodes.

### Exponential Moving Average (EMA) Convolution
EMA is applied to the graph convolution process, improving the stability of predictions by smoothing the feature representations without losing critical information over multiple layers.

---

## 3. Experimental Setup

### Datasets
- **Florida (FL)**: 42 houses, hourly load data, 1 year.
- **California (CA)**: 72 houses, hourly load data, 1 year.
- **Portugal Electricity**: 321 users, hourly load data, 3 years.

The datasets are split chronologically into:
- **Training Set**: 60%
- **Validation Set**: 20%
- **Testing Set**: 20%

### Evaluation Metrics
The following metrics are used to assess the model performance:
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Root Mean Square Error (RMSE)**
- **Root Relative Squared Error (RSE)**
- **Correlation Coefficient (CORR)**

---

## 4. Results

### Comparison with Baselines
The EMAGCN was compared with several baselines including RNN, TCN, LSTNet, and MTGNN. It consistently outperformed these models in terms of accuracy across different time horizons (3h, 6h, 12h, 24h).

#### Results on FL Dataset (42 houses)

| Time Horizon  | MAE    | MAPE   | RMSE   |
|---------------|--------|--------|--------|
| 3 hours       | 0.1017 | 0.0572 | 0.1794 |
| 6 hours       | 0.1327 | 0.0770 | 0.2370 |
| 12 hours      | 0.1569 | 0.0888 | 0.2756 |
| 24 hours      | 0.1705 | 0.0962 | 0.2965 |

#### Results on CA Dataset (72 houses)

| Time Horizon  | MAE    | MAPE   | RMSE   |
|---------------|--------|--------|--------|
| 3 hours       | 0.1162 | 0.0554 | 0.2543 |
| 6 hours       | 0.1501 | 0.0687 | 0.3482 |
| 12 hours      | 0.1800 | 0.0785 | 0.4138 |
| 24 hours      | 0.1839 | 0.0792 | 0.4363 |

#### Results on Portugal Electricity Dataset (321 users)

| Time Horizon  | RSE    | CORR   |
|---------------|--------|--------|
| 3 hours       | 0.0786 | 0.9416 |
| 6 hours       | 0.0889 | 0.9314 |
| 12 hours      | 0.0953 | 0.9210 |
| 24 hours      | 0.0989 | 0.9179 |

### Ablation Study
The ablation study demonstrated that each component of the EMAGCN contributes to the overall performance. Removing any part (static or dynamic graph constructor, EMA-GCN, temporal embeddings) leads to a decrease in accuracy.

| Model Variant     | MAE    | MAPE   | RMSE   |
|-------------------|--------|--------|--------|
| w/o Static GC     | 0.1785 | 0.1019 | 0.3057 |
| w/o Dynamic GC    | 0.1827 | 0.1056 | 0.3064 |
| w/o EMA-GCN       | 0.1815 | 0.1050 | 0.3029 |
| w/o Temporal Emb  | 0.1825 | 0.1040 | 0.3054 |
| Full STEGNN       | 0.1705 | 0.0962 | 0.2965 |

---

## 5. Conclusion

The EMAGCN improves short-term load forecasting by addressing over-smoothing issues in deep GCNs, efficiently capturing both spatial and temporal dependencies in electricity load data. Comparative experiments on multiple datasets demonstrate its superiority over other state-of-the-art methods, making it a valuable contribution to load forecasting research.

