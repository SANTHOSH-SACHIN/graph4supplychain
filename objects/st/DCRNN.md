# DCRNN

Year: 2018  
Publication: ICLR  
Paper Link: https://ieeexplore.ieee.org/document/8845132

#### Task:

- Multistep (M-S)

#### Architecture:

- Discrete (D)
- Continuous (C)

#### Spatial Module:

- Spatial GNN

#### Temporal Module:

- Time Recurrence (T-R)

#### Missing Values:

- No

#### Input Graph:

- Required (R)

#### Learned Graph Relations:

- Not applicable

#### Graph Heuristics:

- Spatial Proximity (SP)

## Links

- https://github.com/liyaguang/DCRNN?tab=readme-ov-file

## 1. Detailed Architecture

### 1.1 Diffusion Convolution (Spatial Modeling)

In DCRNN, spatial dependencies are captured using **diffusion convolution** over a directed graph. This process models the flow of traffic data between nodes (representing sensors), considering both forward and backward directions.

- **Forward Diffusion**: Models how traffic information flows to downstream nodes (forward direction), capturing how congestion at one node impacts future nodes.
- **Backward Diffusion**: Captures the influence of upstream nodes on a given node, allowing the model to account for feedback effects in traffic patterns.

The diffusion process allows the model to consider traffic information from multiple hops away in the graph, thus providing a deeper understanding of spatial dependencies. 

### 1.2 Temporal Modeling (GRU-based)

To model temporal dependencies, DCRNN uses a **Gated Recurrent Unit (GRU)**. In DCRNN, the GRU’s standard linear transformations are replaced by the **diffusion convolution** operations, allowing the model to embed spatial dependencies directly into the recurrent updates.

Key components of GRU include:
- **Reset Gate**: Determines how much of the past information to forget.
- **Update Gate**: Controls how much of the current state is carried forward.
- **Candidate Hidden State**: A proposed hidden state, which is a combination of the input and the previous hidden state.
- **Hidden State Update**: Combines the previous hidden state and the candidate hidden state to update the hidden state.

### 1.3 Encoder-Decoder Framework

DCRNN uses an **encoder-decoder framework** with scheduled sampling:
- **Encoder**: Receives a sequence of traffic data and processes it to learn representations using the diffusion convolutional GRU.
- **Decoder**: Predicts future traffic states by generating sequential outputs based on the learned representations from the encoder. 

**Scheduled Sampling** is employed to gradually replace the ground truth input with the model’s predictions during training, which helps to mitigate the gap between training and testing performance.

---

## 2. Results

### Datasets Used
- **METR-LA**: Traffic sensor data from Los Angeles.
- **PEMS-BAY**: Traffic sensor data from the Bay Area.

The model is evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

### METR-LA Dataset Results

| Horizon  | MAE  | RMSE  | MAPE  |
|----------|------|-------|-------|
| 15 min   | 2.77 | 5.38  | 7.3%  |
| 30 min   | 3.15 | 6.45  | 8.8%  |
| 1 hour   | 3.60 | 7.60  | 10.5% |

### PEMS-BAY Dataset Results

| Horizon  | MAE  | RMSE  | MAPE  |
|----------|------|-------|-------|
| 15 min   | 1.38 | 2.95  | 2.9%  |
| 30 min   | 1.74 | 3.97  | 3.9%  |
| 1 hour   | 2.07 | 4.74  | 4.9%  |

### Comparison with Baselines

DCRNN was compared to multiple baseline models, such as:
- Historical Average (HA)
- Auto-Regressive Integrated Moving Average (ARIMA)
- Support Vector Regression (SVR)

DCRNN consistently outperformed these baselines in terms of prediction accuracy, especially for longer time horizons (30 min and 1 hour).

---

## 3. Conclusion

The **Diffusion Convolutional Recurrent Neural Network (DCRNN)** is highly effective for spatio-temporal traffic forecasting, as it combines **diffusion convolution** for spatial dependencies and **GRU** for temporal dependencies. The **bidirectional diffusion process** enables the model to capture both upstream and downstream traffic flows, improving its ability to predict traffic patterns.

Key strengths of DCRNN include:
- **Spatial and Temporal Integration**: By embedding spatial dependencies directly into the GRU, DCRNN can capture complex interactions between nodes in a traffic network.
- **Handling of Directed Graphs**: The use of directed graphs and diffusion convolution allows the model to handle asymmetric relationships in traffic flows.
- **Scheduled Sampling**: This technique helps the model generalize better by addressing the discrepancy between training and inference.

Overall, DCRNN outperforms traditional baselines, particularly for longer-term predictions, making it a state-of-the-art solution for traffic forecasting and potentially other spatio-temporal prediction tasks.