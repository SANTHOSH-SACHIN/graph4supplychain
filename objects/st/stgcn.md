# STGCN

Year: 2018  
Publication: IJCAI  
Paper Link: https://www.ijcai.org/proceedings/2018/0505.pdf

#### Task:

- Multistep (M-S)

#### Architecture:

- Discrete (D)
- Factorized (F)

#### Spatial Module:

- Spectral GNN

#### Temporal Module:

- Time Convolution (T-C)

#### Missing Values:

- No

#### Input Graph:

- Required (R)

#### Learned Graph Relations:

- Not applicable

#### Graph Heuristics:

- Spatial Proximity (SP)


## 1. Methodology

STGCN integrates spatial and temporal dependencies by combining Graph Convolutional Networks (GCNs) to capture the spatial relations and Temporal Convolutional Networks (TCNs) to capture temporal dynamics. 

- **Spatial Dependencies**: GCN layers capture the relationships between different nodes (e.g., road sensors).
- **Temporal Dependencies**: Temporal convolution layers (TCNs) focus on extracting time-series patterns from traffic data.

## 2. Layers Used

The model architecture consists of:

- **Graph Convolutional Layers**: For learning spatial relations among nodes in a traffic network.
- **Temporal Convolutional Layers (TCN)**: For processing the temporal dependencies of traffic data.
  
## 3. Model Architecture

1. **GCN Layers**: Learn spatial features from the traffic network.
2. **TCN Layers**: Learn temporal features from the traffic data.
3. **Fully Connected Layer**: For final traffic flow prediction.

## 4. Hyperparameters

- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Dropout Rate**: 0.3
- **Epochs**: 200
- **Optimizer**: Adam

## 5. Performance Metrics

| Dataset   | Time   | Metric | ARIMA  | GCN   | GRU   | STGCN     |
|-----------|--------|--------|--------|-------|-------|-----------|
| PeMSD7    | 15 min | MAE    | 5.10   | 4.80  | 4.65  | **4.21**  |
|           | 30 min | MAE    | 5.40   | 5.10  | 4.95  | **4.35**  |
|           | 60 min | MAE    | 5.90   | 5.60  | 5.40  | **4.50**  |
|           | 15 min | RMSE   | 8.10   | 7.50  | 7.35  | **6.90**  |
|           | 30 min | RMSE   | 8.60   | 8.10  | 7.80  | **7.35**  |
|           | 60 min | RMSE   | 9.20   | 8.90  | 8.60  | **7.90**  |