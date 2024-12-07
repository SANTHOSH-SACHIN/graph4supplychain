# GraphSAGE: Inductive Representation Learning on Large Graphs

Year: 2018  
Publication: NIPS Conference  
Paper Link: https://arxiv.org/abs/1706.02216

#### Task:

- Node Classification (Inductive Learning)

#### Architecture:

- Graph SAmple and aggreGatE (GraphSAGE)

#### Spatial Module:

- Aggregator Functions (Mean, LSTM, Pooling, GCN)

#### Temporal Module:

- Not applicable (focus on spatial relationships via aggregation)

#### Missing Values:

- No

#### Input Graph:

- Required (Node Features and Neighborhood Information)

#### Learned Graph Relations:

- Dynamic (via trainable aggregation functions)

#### Graph Heuristics:

- Multi-hop Neighborhood Sampling

---

# GraphSAGE: Inductive Representation Learning on Large Graphs

## 1. Introduction

GraphSAGE is an inductive framework that generates embeddings for nodes in graphs. Instead of learning a separate embedding for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from the local neighborhood of a node. This approach generalizes to unseen nodes, making it ideal for dynamic graphs where new nodes are constantly added.

### Key Highlights:

- Combines node feature information with structural information from neighborhoods.
- Efficiently generalizes to unseen nodes and unseen graphs.
- Multiple aggregation methods (Mean, LSTM, Pooling) provide flexibility in capturing neighborhood information.

## 2. Methodology

The model is designed to:

- **Neighborhood Aggregation**: Aggregates features from the local neighborhood of a node using trainable aggregator functions.
- **Inductive Learning**: Generalizes to unseen nodes and unseen graphs without requiring retraining.

### Additional Details:

- **Aggregation Functions**: GraphSAGE offers different types of aggregation functions: Mean, LSTM, Pooling, and GCN.
- **Inductive Learning**: Can be applied to new nodes without retraining the model, unlike transductive methods such as DeepWalk or GCN.

## 3. Model Components

The **GraphSAGE** model includes:

- **Aggregator Functions**: Trainable functions (Mean, LSTM, Pooling) used to aggregate neighborhood information.
- **Multi-hop Neighborhood Sampling**: Samples nodes from multiple hops away to capture long-range dependencies.

## 4. Hyperparameters

- **Learning Rate**: 0.01 (default)
- **Optimizer**: Adam
- **Neighborhood Sample Sizes**: 25 (1-hop neighbors), 10 (2-hop neighbors)
- **Evaluation Metrics**: F1-score (for multi-label classification)
- **Batch Size**: 512

## 5. Performance Metrics

| Method                     | Citation Unsup. F1 | Citation Sup. F1 | Reddit Unsup. F1 | Reddit Sup. F1 | PPI Unsup. F1 | PPI Sup. F1 |
|-----------------------------|--------------------|-------------------|-------------------|-----------------|---------------|-------------|
| Random                      | 0.206              | 0.206             | 0.043             | 0.042           | 0.396         | 0.396       |
| Raw Features                 | 0.575              | 0.575             | 0.585             | 0.585           | 0.422         | 0.422       |
| DeepWalk                     | 0.565              | 0.565             | 0.324             | 0.324           | —             | —           |
| DeepWalk + Features          | 0.701              | 0.701             | 0.691             | 0.691           | —             | —           |
| GraphSAGE-GCN                | 0.742              | 0.772             | 0.908             | 0.930           | 0.465         | 0.500       |
| GraphSAGE-Mean               | 0.778              | 0.820             | 0.897             | 0.950           | 0.486         | 0.598       |
| GraphSAGE-LSTM               | 0.788              | 0.832             | 0.907             | 0.954           | 0.482         | 0.612       |
| GraphSAGE-Pool               | 0.798              | 0.839             | 0.892             | 0.948           | 0.502         | 0.600       |

This table highlights how GraphSAGE achieves or surpasses state-of-the-art performance across multiple datasets (Citation, Reddit, PPI) using different aggregator functions.

## 6. Experimental Setup

- **Datasets**: Experiments were performed on three datasets: Citation (Web of Science), Reddit (post classification), and Protein-Protein Interaction (PPI) graphs.
- **Model Setup**: The GraphSAGE model was evaluated in both unsupervised and supervised learning settings. Different variants (Mean, LSTM, Pooling, GCN) of the aggregator functions were tested.
- **Batching**: All models used a minibatch size of 512, except for DeepWalk, which used a batch size of 64.

## 7. Sensitivity Analysis

- **Neighborhood Size**: Increasing the neighborhood sample size improves performance but also increases runtime.
- **Aggregator Choice**: The Pooling and LSTM aggregators generally performed best, with the Pooling aggregator being the most efficient in terms of runtime.

## 8. Conclusion

GraphSAGE is a powerful framework for inductive representation learning on large graphs. By leveraging node features and trainable aggregation functions, it outperforms transductive methods on unseen nodes and graphs.

Future work includes exploring non-uniform neighborhood sampling techniques and extending the framework to handle multi-modal graphs.
