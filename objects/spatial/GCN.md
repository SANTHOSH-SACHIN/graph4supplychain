# N-GCN: Multi-scale Graph Convolution for Semi-supervised Node Classification

Year: 2018  
Publication: Preprint on arXiv  
Paper Link: https://arxiv.org/abs/1802.08888

#### Task:

- Semi-supervised Node Classification

#### Architecture:

- Network of Graph Convolutional Networks (N-GCN)

#### Spatial Module:

- Graph Convolutional Network (GCN)

#### Temporal Module:

- Not applicable (focus is on spatial relationships via graph convolution)

#### Missing Values:

- No

#### Input Graph:

- Required (Adjacency Matrix and Node Features)

#### Learned Graph Relations:

- Multi-hop connections (via powers of the adjacency matrix)

#### Graph Heuristics:

- Random Walks, Power of Adjacency Matrix

---

# N-GCN: Multi-scale Graph Convolution for Semi-supervised Node Classification

## 1. Introduction

The **N-GCN** model leverages multi-hop node connections and combines multiple GCN layers applied to different powers of the adjacency matrix. The model uses random walk statistics to enhance semi-supervised learning on graph data and improves classification by learning over multiple scales of the graph.

### Key Highlights:

- Combines GCN modules applied to varying powers of the adjacency matrix.
- Effectively generalizes across different datasets (Cora, Citeseer, Pubmed, and PPI).
- Robust to adversarial input perturbations by utilizing higher-order adjacency matrices.
- Works well with few labeled nodes in semi-supervised node classification tasks.

## 2. Methodology

The model is designed to:

- **Spatial Dependency**: Captures node features and relationships over multiple graph scales using powers of the normalized adjacency matrix.
- **Multi-GCN Modules**: Each GCN module processes a different power of the adjacency matrix (e.g., \(A^0\), \(A^1\), ..., \(A^k\)).
- **Classification Sub-network**: Combines the outputs of these GCNs for final classification.

### Additional Details:

- **Random Walks**: The model incorporates random walk statistics to enhance the learning capability over long-range node connections.
- **Attention Mechanism**: N-GCN can be extended with attention mechanisms to selectively weight different GCN outputs, improving feature selection across varying scales.

## 3. Model Components

The **N-GCN** model includes:

- **Graph Convolution Layers**: Applies multiple GCN layers to capture different scales of node information.
- **Classification Sub-network**: Fully-connected layer that combines the output of all GCNs for final prediction.
- **Replication Factor**: Multiple GCNs are instantiated to further improve performance and capture more diverse node relationships.

## 4. Hyperparameters

- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **GCN Layers**: 2 layers per GCN module
- **Replication Factor**: 4 (default)
- **Powers of Adjacency Matrix (K)**: Typically up to 5
- **Evaluation Metrics**: Accuracy (for citation datasets), F1-score (for PPI)
- **Regularization**: L2 regularization with \( \lambda = 10^{-5} \)
- **Dropout**: 50% dropout is used to prevent overfitting.

## 5. Performance Metrics

# Node Classification Performance

| Method                     | Citeseer | Cora  | Pubmed | PPI   |
|-----------------------------|----------|-------|--------|-------|
| (a) ManiReg [7]             | 60.1     | 59.5  | 70.7   | –     |
| (b) SemiEmb [24]            | 59.6     | 59.0  | 71.1   | –     |
| (c) LP [26]                 | 45.3     | 68.0  | 63.0   | –     |
| (d) DeepWalk [22]           | 43.2     | 67.2  | 65.3   | –     |
| (e) ICA [18]                | 69.1     | 75.1  | 73.9   | –     |
| (f) Planetoid [25]          | 64.7     | 75.7  | 77.2   | –     |
| (g) GCN [14]                | 70.3     | 81.5  | 79.0   | –     |
| (h) SAGE-LSTM [11]          | –        | –     | –      | 61.2  |
| (i) SAGE [11]               | –        | –     | –      | 60.0  |
| (j) DCNN (our implementation)| 71.1     | 81.3  | 79.3   | 44.0  |
| (k) GCN (our implementation) | 71.2     | 81.0  | 78.8   | 46.2  |
| (l) SAGE (our implementation)| 63.5     | 77.4  | 77.6   | 59.8  |
| (m) N-GCN (ours)            | 72.2     | 83.0  | 79.5   | 46.8  |
| (n) N-SAGE (ours)           | 71.0     | 81.8  | 79.4   | 65.0  |


| Dataset  | Citeseer | Cora   | Pubmed | PPI    |
|----------|----------|--------|--------|--------|
| N-GCN    | 72.2%    | 83.0%  | 79.5%  | 46.8%  |
| Baseline | 70.3%    | 81.5%  | 79.0%  | 46.2%  |

These datasets include citation graphs and biological graphs with node features and classification labels.

## 6. Experimental Setup

- **Datasets**: The experiments were performed on citation datasets (Citeseer, Cora, Pubmed) and a biological Protein-Protein Interaction (PPI) dataset.
- **Data Splits**: The datasets follow standard data splits: 20 nodes per class for training, 500 nodes for validation, and 1000 nodes for testing.
- **Comparison**: N-GCN outperforms other methods such as Planetoid, DeepWalk, and vanilla GCN across various datasets.
- **Feature Noise Resilience**: N-GCN demonstrates resilience to input noise by effectively shifting attention to nodes with less noisy features.

## 7. Sensitivity Analysis

- **Impact of Random Walk Length (K)**: Model accuracy improves as the random walk step (K) increases, with diminishing returns after a certain length.
- **Impact of Replication Factor (r)**: Increasing the replication factor generally boosts performance by increasing the model's capacity to capture diverse node relationships.
- **Tolerance to Feature Noise**: N-GCN performs well under feature removal or noise, showcasing robustness in practical scenarios where input data may be incomplete.

## 8. Conclusion

The N-GCN model enhances the capability of traditional GCNs by combining multiple scales of graph structure through random walks and adjacency powers. This model sets new state-of-the-art performance on multiple node classification tasks and can generalize well to other graph-based architectures like GraphSAGE.

Future work includes extending this model to larger datasets and incorporating stochastic training mechanisms for further scalability.
