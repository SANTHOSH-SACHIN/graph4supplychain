# GAT: Graph Attention Networks

Year: 2018  
Publication: ICLR Conference  
Paper Link: https://arxiv.org/abs/1710.10903

#### Task:

- Node Classification (Transductive and Inductive Learning)

#### Architecture:

- Graph Attention Network (GAT)

#### Spatial Module:

- Graph Attentional Layer

#### Temporal Module:

- Not applicable (focus on spatial relationships using attention mechanism)

#### Missing Values:

- No

#### Input Graph:

- Required (Node Features and Adjacency Matrix)

#### Learned Graph Relations:

- Dynamic (via attention mechanism)

#### Graph Heuristics:

- Self-attention, Multi-head attention

---

# GAT: Graph Attention Networks

## 1. Introduction

The **GAT** model introduces attention mechanisms into graph neural networks, allowing nodes to weigh the importance of their neighbors dynamically. This model addresses limitations of prior GCN models that used fixed weighting based on the adjacency matrix.

### Key Highlights:

- Leverages **masked self-attention** to dynamically learn the importance of neighboring nodes.
- Computationally efficient—no need for matrix decompositions or knowing the full graph structure upfront.
- Suitable for both **transductive** and **inductive** learning tasks.

## 2. Methodology

The model is designed to:

- **Attention Mechanism**: GAT uses attention to compute dynamic importance scores between nodes, applying different weights to each neighbor.
- **Multi-head Attention**: The model benefits from the use of multiple attention heads to stabilize the learning process and enrich feature extraction.
- **Graph Attention Layer**: The core GAT layer computes attention scores for node pairs and aggregates their features based on these scores.

### Additional Details:

- **Dynamic Neighborhood Weights**: Each node's neighbors can be assigned different importance based on learned attention coefficients.
- **Multi-head Attention**: Several attention mechanisms (heads) are used in parallel, and their results are either concatenated or averaged for final output.
- **No Fixed Graph Structure Required**: The attention mechanism works without requiring the full adjacency matrix.

## 3. Model Components

The **GAT** model includes:

- **Graph Attentional Layers**: Uses attention mechanisms to learn the importance of neighboring nodes and aggregate their features.
- **Multi-head Attention**: Multiple attention heads compute features independently, and their outputs are either concatenated or averaged.
- **Non-linearity**: After the attention coefficients are computed, the weighted feature sum passes through a non-linearity such as **LeakyReLU** or **ELU**.

## 4. Hyperparameters

- **Learning Rate**: 0.005 (for Cora, Citeseer) and 0.01 (for Pubmed)
- **Optimizer**: Adam
- **Attention Heads**: 
  - 8 heads in the hidden layers
  - 1 head for the final output layer (for transductive tasks)
  - 6 heads for multi-label classification (for inductive tasks)
- **Evaluation Metrics**: Accuracy (for citation networks), F1-Score (for PPI dataset)
- **Dropout**: 0.6 applied to input layers and attention coefficients
- **L2 Regularization**: 0.0005 for Cora and Citeseer, 0.001 for Pubmed

## 5. Performance Metrics

| Method     | Cora           | Citeseer       | Pubmed         |
|------------|----------------|----------------|----------------|
| MLP        | 55.1%          | 46.5%          | 71.4%          |
| ManiReg    | 59.5%          | 60.1%          | 70.7%          |
| SemiEmb    | 59.0%          | 59.6%          | 71.7%          |
| LP         | 68.0%          | 45.3%          | 63.0%          |
| DeepWalk   | 67.2%          | 43.2%          | 65.3%          |
| ICA        | 75.1%          | 69.1%          | 73.9%          |
| Planetoid  | 75.7%          | 64.7%          | 77.2%          |
| Chebyshev  | 81.2%          | 69.8%          | 74.4%          |
| GCN        | 81.5%          | 70.3%          | 79.0%          |
| MoNet      | 81.7 ± 0.5%    | —              | 78.8 ± 0.3%    |
| GCN-64∗    | 81.4 ± 0.5%    | 70.9 ± 0.5%    | 79.0 ± 0.3%    |
| **GAT**    | **83.0 ± 0.7%** | **72.5 ± 0.7%** | **79.0 ± 0.3%** |


| Dataset  | Cora   | Citeseer | Pubmed | PPI    |
|----------|--------|----------|--------|--------|
| GAT      | 83.0%  | 72.5%    | 79.0%  | 97.3%  |
| Baseline | 81.5%  | 70.3%    | 79.0%  | 61.2%  |

This table highlights how **GAT** achieves or surpasses state-of-the-art performance across multiple datasets (Cora, Citeseer, Pubmed) by using the attention mechanism to assign different weights to neighboring nodes.

## 6. Experimental Setup

- **Datasets**: Experiments were performed on citation network datasets (Cora, Citeseer, Pubmed) and the Protein-Protein Interaction (PPI) dataset.
- **Model Setup**: For transductive tasks, a two-layer GAT was used with 8 attention heads in the hidden layer. For inductive tasks, a three-layer GAT was used with 4 attention heads in the hidden layers.
- **Batching**: Inductive tasks used a batch size of 2 graphs. Glorot initialization and Adam optimizer were used, with early stopping based on validation accuracy.

## 7. Sensitivity Analysis

- **Multi-head Attention**: Using more attention heads (e.g., 8 in the hidden layers) improves stability and performance.
- **Dropout and L2 Regularization**: Helps prevent overfitting, especially when training on smaller datasets like Cora and Citeseer.
- **Neighborhood Size**: GAT does not rely on fixed neighborhood sizes, which is an advantage over other models that require predefined neighborhood sizes.

## 8. Conclusion

The **GAT** model is a powerful extension to graph-based neural networks, allowing for dynamic and flexible learning of node importance through self-attention. It achieves state-of-the-art results in both transductive and inductive node classification tasks.

Future work includes exploring how GAT can be extended to graph-level tasks, such as graph classification, and how to incorporate edge features into the attention mechanism.
