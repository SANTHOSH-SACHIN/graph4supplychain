# import os
# import zipfile
# import requests
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.preprocessing import StandardScaler
# import h5py
# import pickle
# import scipy.sparse as sp
# from kaggle.api.kaggle_api_extended import KaggleApi

# # Import the temporal signal class from torch_geometric_temporal
# from torch_geometric_temporal.signal import StaticGraphTemporalSignal
# from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv

# # --------------------------
# # 1. Dataset Handling with StaticGraphTemporalSignal
# # --------------------------
# def load_dataset(dataset_name):
#     if dataset_name == 'PeMSD7':
#         # Load speed measurements and graph connectivity
#         speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None).values  # shape (T, N)
#         adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
#         edge_weight = None  # no edge attributes provided
#     elif dataset_name == 'METR-LA':
#         # Load speed measurements
#         with h5py.File('dataset/metr-la.h5', 'r') as f:
#             speed_data = pd.DataFrame(f['df']['block0_values'][:]).values
#         # Load adjacency matrix (we assume the third element contains it)
#         with open('dataset/adj_mx.pkl', 'rb') as f:
#             adj_mx = pickle.load(f, encoding='latin1')
#         adjacency_matrix = adj_mx[2]
#         # For METR-LA, create edge_weight from the sparse matrix
#         adj_sparse = sp.coo_matrix(adjacency_matrix)
#         edge_weight = torch.tensor(adj_sparse.data, dtype=torch.float)
#     else:
#         raise ValueError("Dataset name must be either 'PeMSD7' or 'METR-LA'.")

#     # Standardize the speed data (each column/sensor is normalized)
#     scaler = StandardScaler()
#     speed_data = scaler.fit_transform(speed_data)

#     # Build edge_index from the adjacency matrix (assumes nonzero entries indicate edges)
#     if dataset_name == 'PeMSD7':
#         edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
#     else:
#         # For METR-LA we already created a sparse representation above:
#         adj_sparse = sp.coo_matrix(adjacency_matrix)
#         edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)

#     # Create temporal snapshots:
#     # For a time series of shape (T, N) we use each time step as a snapshot input
#     # and the next time step as the target. We also unsqueeze so that each snapshot is [N, 1].
#     features = [torch.tensor(speed_data[t], dtype=torch.float).unsqueeze(1)
#                 for t in range(speed_data.shape[0] - 1)]
#     targets  = [torch.tensor(speed_data[t+1], dtype=torch.float).unsqueeze(1)
#                 for t in range(speed_data.shape[0] - 1)]

#     return StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)

# # --------------------------
# # 2. Model Definition
# # --------------------------
# class GNNModel(nn.Module):
#     def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_weight=None, encoder_type='Transformer'):
#         super().__init__()
#         self.edge_index = edge_index
#         self.edge_weight = edge_weight

#         # Select encoder type
#         if encoder_type == 'SAGE':
#             self.encoder = nn.Sequential(
#                 SAGEConv((-1, -1), hidden_channels),
#                 nn.ReLU(),
#                 SAGEConv((-1, -1), hidden_channels)
#             )
#         elif encoder_type == 'GAT':
#             self.encoder = nn.Sequential(
#                 GATConv((-1, -1), hidden_channels, add_self_loops=False),
#                 nn.ReLU(),
#                 GATConv((-1, -1), hidden_channels, add_self_loops=False)
#             )
#         elif encoder_type == 'General':
#             self.encoder = nn.Sequential(
#                 GeneralConv((-1, -1), hidden_channels),
#                 nn.ReLU(),
#                 GeneralConv((-1, -1), hidden_channels)
#             )
#         elif encoder_type == 'Transformer':
#             # Note: edge_dim is not used here, so we pass None
#             self.encoder = nn.Sequential(
#                 TransformerConv(input_size, hidden_channels, edge_dim=None),
#                 nn.ReLU(),
#                 TransformerConv(hidden_channels, hidden_channels, edge_dim=None)
#             )
#         else:
#             raise ValueError(f"Unknown encoder type: {encoder_type}")

#         # Simple MLP decoder that maps the encoded node embeddings to the target signal
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_channels, out_channels)
#         )

#     def forward(self, x):
#         # x: node features of shape [num_nodes, input_size] for a given time snapshot
#         if isinstance(self.encoder[0], TransformerConv):
#             x = self.encoder[0](x, self.edge_index, self.edge_weight)
#             x = self.encoder[1](x)
#             x = self.encoder[2](x, self.edge_index, self.edge_weight)
#         else:
#             x = self.encoder[0](x, self.edge_index)
#             x = self.encoder[1](x)
#             x = self.encoder[2](x, self.edge_index)
#         return self.decoder(x)

# # --------------------------
# # 3. Training & Evaluation
# # --------------------------
# def train_evaluate(dataset_name, encoder_type='Transformer', epochs=50, lr=0.001):
#     # Load the dataset as a temporal signal
#     signal = load_dataset(dataset_name)
#     # Split the temporal snapshots into training (80%) and testing (20%)
#     num_snapshots = len(signal.features)
#     train_size = int(num_snapshots * 0.8)
#     train_features = signal.features[:train_size]
#     train_targets  = signal.targets[:train_size]
#     test_features  = signal.features[train_size:]
#     test_targets   = signal.targets[train_size:]

#     # Create separate StaticGraphTemporalSignal objects for train and test splits
#     train_signal = StaticGraphTemporalSignal(signal.edge_index, signal.edge_weight, train_features, train_targets)
#     test_signal  = StaticGraphTemporalSignal(signal.edge_index, signal.edge_weight, test_features, test_targets)

#     # Input size is the number of features per node (should be 1 after unsqueeze)
#     input_size = train_features[0].shape[1]
#     model = GNNModel(input_size=input_size,
#                      hidden_channels=64,
#                      out_channels=input_size,
#                      edge_index=signal.edge_index,
#                      edge_weight=signal.edge_weight,
#                      encoder_type=encoder_type)
#     optimizer = Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     # Training loop: iterate over each temporal snapshot in the training signal
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for snapshot, target in zip(train_signal.features, train_signal.targets):
#             output = model(snapshot)
#             loss = loss_fn(output, target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_signal.features):.4f}")

#     # Evaluation on the test set
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for snapshot, target in zip(test_signal.features, test_signal.targets):
#             output = model(snapshot)
#             loss = loss_fn(output, target)
#             total_loss += loss.item()
#     print(f"Test Loss: {total_loss/len(test_signal.features):.4f}")
#     return model


# if __name__ == "__main__":
#     # For PeMSD7 experiments
#     train_evaluate('PeMSD7', encoder_type='GAT')
#     train_evaluate('PeMSD7', encoder_type='SAGE')
#     train_evaluate('PeMSD7', encoder_type='General')
#     train_evaluate('PeMSD7', encoder_type='Transformer')

#     # For METR-LA experiments
#     train_evaluate('METR-LA', encoder_type='GAT')
#     train_evaluate('METR-LA', encoder_type='SAGE')
#     train_evaluate('METR-LA', encoder_type='General')
#     train_evaluate('METR-LA', encoder_type='Transformer')


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import h5py
import pickle
import scipy.sparse as sp
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv

# --------------------------
# 1. Dataset Handling with StaticGraphTemporalSignal
# --------------------------
def load_dataset(dataset_name):
    if dataset_name == 'PeMSD7':
        # Load speed measurements and graph connectivity
        speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None).values  # shape (T, N)
        adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
        edge_weight = None  # no edge attributes provided
    elif dataset_name == 'METR-LA':
        # Load speed measurements
        with h5py.File('dataset/metr-la.h5', 'r') as f:
            speed_data = pd.DataFrame(f['df']['block0_values'][:]).values
        # Load adjacency matrix (we assume the third element contains it)
        with open('dataset/adj_mx.pkl', 'rb') as f:
            adj_mx = pickle.load(f, encoding='latin1')
        adjacency_matrix = adj_mx[2]
        # For METR-LA, create edge_weight from the sparse matrix
        adj_sparse = sp.coo_matrix(adjacency_matrix)
        edge_weight = torch.tensor(adj_sparse.data, dtype=torch.float)
    elif dataset_name == 'Chickenpox':
        # Load Chickenpox data
        speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
        adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
        edge_weight = None  # no edge attributes provided

        # Remove non-numeric columns (e.g., dates)
        speed_data = speed_data.select_dtypes(include=[np.number]).values
    else:
        raise ValueError("Dataset name must be either 'PeMSD7', 'METR-LA', or 'Chickenpox'.")

    # Standardize the speed data (each column/sensor is normalized)
    scaler = StandardScaler()
    speed_data = scaler.fit_transform(speed_data)

    # Build edge_index from the adjacency matrix (assumes nonzero entries indicate edges)
    if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
        edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
    else:
        # For METR-LA we already created a sparse representation above:
        adj_sparse = sp.coo_matrix(adjacency_matrix)
        edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)

    # Create temporal snapshots:
    # For a time series of shape (T, N) we use each time step as a snapshot input
    # and the next time step as the target. We also unsqueeze so that each snapshot is [N, 1].
    features = [torch.tensor(speed_data[t], dtype=torch.float).unsqueeze(1)
                for t in range(speed_data.shape[0] - 1)]
    targets  = [torch.tensor(speed_data[t+1], dtype=torch.float).unsqueeze(1)
                for t in range(speed_data.shape[0] - 1)]

    return StaticGraphTemporalSignal(edge_index, edge_weight, features, targets)

# --------------------------
# 2. Model Definition
# --------------------------
class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_weight=None, encoder_type='Transformer'):
        super().__init__()
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # Select encoder type
        if encoder_type == 'SAGE':
            self.encoder = nn.Sequential(
                SAGEConv((-1, -1), hidden_channels),
                nn.ReLU(),
                SAGEConv((-1, -1), hidden_channels)
            )
        elif encoder_type == 'GAT':
            self.encoder = nn.Sequential(
                GATConv((-1, -1), hidden_channels, add_self_loops=False),
                nn.ReLU(),
                GATConv((-1, -1), hidden_channels, add_self_loops=False)
            )
        elif encoder_type == 'General':
            self.encoder = nn.Sequential(
                GeneralConv((-1, -1), hidden_channels),
                nn.ReLU(),
                GeneralConv((-1, -1), hidden_channels)
            )
        elif encoder_type == 'Transformer':
            # Note: edge_dim is not used here, so we pass None
            self.encoder = nn.Sequential(
                TransformerConv(input_size, hidden_channels, edge_dim=None),
                nn.ReLU(),
                TransformerConv(hidden_channels, hidden_channels, edge_dim=None)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Simple MLP decoder that maps the encoded node embeddings to the target signal
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        # x: node features of shape [num_nodes, input_size] for a given time snapshot
        if isinstance(self.encoder[0], TransformerConv):
            x = self.encoder[0](x, self.edge_index, self.edge_weight)
            x = self.encoder[1](x)
            x = self.encoder[2](x, self.edge_index, self.edge_weight)
        else:
            x = self.encoder[0](x, self.edge_index)
            x = self.encoder[1](x)
            x = self.encoder[2](x, self.edge_index)
        return self.decoder(x)

# --------------------------
# 3. Training & Evaluation
# --------------------------
def train_evaluate(dataset_name, encoder_type='Transformer', epochs=50, lr=0.001):
    # Load the dataset as a temporal signal
    signal = load_dataset(dataset_name)
    # Split the temporal snapshots into training (80%) and testing (20%)
    num_snapshots = len(signal.features)
    train_size = int(num_snapshots * 0.8)
    train_features = signal.features[:train_size]
    train_targets  = signal.targets[:train_size]
    test_features  = signal.features[train_size:]
    test_targets   = signal.targets[train_size:]

    # Create separate StaticGraphTemporalSignal objects for train and test splits
    train_signal = StaticGraphTemporalSignal(signal.edge_index, signal.edge_weight, train_features, train_targets)
    test_signal  = StaticGraphTemporalSignal(signal.edge_index, signal.edge_weight, test_features, test_targets)

    # Input size is the number of features per node (should be 1 after unsqueeze)
    input_size = train_features[0].shape[1]
    model = GNNModel(input_size=input_size,
                     hidden_channels=64,
                     out_channels=input_size,
                     edge_index=signal.edge_index,
                     edge_weight=signal.edge_weight,
                     encoder_type=encoder_type)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training loop: iterate over each temporal snapshot in the training signal
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for snapshot, target in zip(train_signal.features, train_signal.targets):
            output = model(snapshot)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_signal.features):.4f}")

    # Evaluation on the test set
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for snapshot, target in zip(test_signal.features, test_signal.targets):
            output = model(snapshot)
            loss = loss_fn(output, target)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(test_signal.features):.4f}")
    return model

if __name__ == "__main__":
    # For PeMSD7 experiments
    train_evaluate('PeMSD7', encoder_type='GAT')
    train_evaluate('PeMSD7', encoder_type='SAGE')
    train_evaluate('PeMSD7', encoder_type='General')
    train_evaluate('PeMSD7', encoder_type='Transformer')

    # For METR-LA experiments
    train_evaluate('METR-LA', encoder_type='GAT')
    train_evaluate('METR-LA', encoder_type='SAGE')
    train_evaluate('METR-LA', encoder_type='General')
    train_evaluate('METR-LA', encoder_type='Transformer')

    # For Chickenpox experiments
    train_evaluate('Chickenpox', encoder_type='GAT')
    train_evaluate('Chickenpox', encoder_type='SAGE')
    train_evaluate('Chickenpox', encoder_type='General')
    train_evaluate('Chickenpox', encoder_type='Transformer')
