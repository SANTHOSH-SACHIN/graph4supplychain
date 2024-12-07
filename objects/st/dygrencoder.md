# Dynamic Graph Encoder (DyGrEncoder)

*Year*: 2019  
*Publication*: IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining  
*Paper Link*: [doi.org/10.1145/3341161.3342872](http://dx.doi.org/10.1145/3341161.3342872)  

#### Task:
- *Dynamic Graph Representation Learning*

#### Architecture:
- *Dynamic Graph Autoencoder (DyGrAE)* and *Dynamic Graph Predictor (DyGrPr)*

#### Spatial Module:
- *Gated Graph Neural Networks (GGNNs)* for learning topological structure

#### Temporal Module:
- *LSTM Encoder-Decoder* for temporal dynamics, with an *attention mechanism* for DyGrPr

#### Missing Values:
- *Handled via LSTM Encoder*

#### Input Graph:
- *Required* as dynamic graph snapshots

#### Learned Graph Relations:
- *Dynamic* (learned temporal relations across snapshots)

#### Graph Heuristics:
- *Attention Mechanism* for identifying impactful time steps, *Temporal Dependencies* through LSTM Encoder-Decoder

## 1. Introduction
DyGEncoder addresses dynamic graph representation by combining GGNNs with LSTM and attention mechanisms. This unsupervised learning approach targets capturing topological and temporal dynamics, allowing for more informative representations suited to tasks like dynamic graph classification.

## 2. Methodology
The model includes:
- *Spatial Dependency*: GGNN captures topology at each time step.
- *Temporal Dependency*: LSTM Encoder for short-term dependencies, attention in DyGrPr for impactful historical steps in prediction.
- *Dynamic Events*: Captures graph dynamics across time with recurrent and attention mechanisms for prediction.

## 3. Model Components
- *Spatial Aggregation*: GGNNs handle node interactions per snapshot.
- *Temporal Aggregation*: LSTM Encoder for sequence modeling; attention (DyGrPr) emphasizes impactful historical steps.
- *Attention Mechanism*: In DyGrPr, weighted attention over time selects relevant steps for future predictions.

## 4. Hyperparameters
- *Learning Rate*: 0.001 (Adam optimizer)
- *Hidden Dimensions*: 100 (GGNN, LSTM)
- *Batch Size*: 50
- *Evaluation Metrics*: Accuracy (Classification Task)

## 5. Performance Metrics

| Dataset            | Day 1 Accuracy | Day 2 Accuracy |
|--------------------|----------------|----------------|
| Baboon Behavior    | 89.91%        | 89.30%        |
| Brain Networks (MGB) | 81.14%       | N/A           |

Two datasets, Baboon Behavior and Brain Networks, evaluated model classification accuracy, highlighting DyGrPr’s advantage in temporal dynamics for enhanced predictive power.

**Dataset**: Chickenpox

DyGrEncoder is designed to dynamically encode evolving graphs for forecasting tasks by capturing both spatial and temporal structures.

## Results on Chickenpox Dataset:

- **MSE**: 0.9655
- **MAE**: 0.6355
- **MAPE**: 1060.5135%
- **RMSE**: 0.9826
- **R-squared**: 0.0411

## Version History:

- v1.0
