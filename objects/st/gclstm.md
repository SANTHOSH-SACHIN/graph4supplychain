# Graph Convolutional LSTM (GCLSTM)

*Year*: 2021  
*Publication*: Not specified  
*Paper Link*: Not provided  

#### Task:
- *Dynamic Network Link Prediction*

#### Architecture:
- *GC-LSTM*: Combines Graph Convolution Network (GCN) with LSTM layers

#### Spatial Module:
- *Graph Convolutional Network (GCN)* to capture structural features in each snapshot

#### Temporal Module:
- *LSTM* for temporal sequence learning with hidden and cell states, utilizing GCN embeddings

#### Missing Values:
- *Handled by integrating GCN into LSTM cell states*

#### Input Graph:
- *Required* as a series of dynamic graph snapshots

#### Learned Graph Relations:
- *Dynamic*, evolving across snapshots

#### Graph Heuristics:
- *Uses K-hop GCN* for spatial context and *LSTM* to manage long-term dependencies

## 1. Introduction
GC-LSTM targets dynamic network link prediction by embedding GCN in LSTM, which allows the model to capture structural changes over time. It can predict both newly formed and disappearing links, supporting real-time network updates in fields like social networks and bioinformatics.

## 2. Methodology
The model includes:
- *Spatial Dependency*: GCN captures node relationships within snapshots.
- *Temporal Dependency*: LSTM tracks temporal patterns across snapshots, supported by hidden and cell states.
- *Network Evolution*: GCN embedded in LSTM enables prediction of link dynamics across time.

## 3. Model Components
- *Spatial Aggregation*: GCN learns structural dependencies within each snapshot.
- *Temporal Aggregation*: LSTM captures link dynamics over multiple snapshots.
- *Hybrid Structure*: GCN embedded into LSTM cells for joint spatio-temporal learning.

## 4. Hyperparameters
- *Learning Rate*: 0.001 (Adam optimizer)
- *Hidden Dimensions*: 256 (first datasets); 512 (larger datasets)
- *Batch Size*: Not specified
- *Evaluation Metrics*: AUC, GMAUC, Error Rate (ER), ER+, ER-

## 5. Performance Metrics

| Dataset          | AUC (20 Samples) | AUC (80 Samples) | GMAUC (20 Samples) | GMAUC (80 Samples) | ER | ER+ | ER- |
|------------------|------------------|------------------|---------------------|---------------------|----|-----|-----|
| CONTACT          | 0.9649           | 0.9453          | 0.9680             | 0.9456             | 0.2324 | 0.1703 | 0.0621 |
| HYPERTEXT09      | 0.9573           | 0.9675          | 0.9921             | 0.9788             | 0.1988 | 0.1902 | 0.0084 |
| ENRON            | 0.8425           | 0.8135          | 0.8571             | 0.8134             | 0.3763 | 0.3070 | 0.0857 |
| RADOSLAW         | 0.9838           | 0.9833          | 0.9980             | 0.9983             | 0.1783 | 0.1008 | 0.0704 |
| FB-FORUM         | 0.9021           | 0.9098          | 0.9062             | 0.9123             | 0.2702 | 0.2186 | 0.0515 |
| LKML             | 0.9082           | 0.9000          | 0.9184             | 0.9049             | 0.7369 | 0.6198 | 0.1171 |

The GC-LSTM model outperformed other baselines, especially in Error Rate metrics, showing its ability to effectively handle dynamic network evolution with both spatial and temporal dependencies.

**Dataset**: Chickenpox

GCLSTM integrates graph convolutional layers into LSTMs to model spatial-temporal dependencies in graph-structured time series data.

## Results on Chickenpox Dataset:

- **MSE**: 1.1536
- **MAE**: 0.6915
- **R-squared**: -0.0925

## Version History:

- v1.0
