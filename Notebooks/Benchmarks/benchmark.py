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
# from torch_geometric.data import Data
# from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv
# import h5py
# import pickle
# import scipy.sparse as sp
# from kaggle.api.kaggle_api_extended import KaggleApi

# # --------------------------
# # 1. Dataset Handling
# # --------------------------
# def load_dataset(dataset_name):
#     if dataset_name == 'PeMSD7':
#         # Download PeMSD7 if missing

#         speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
#         adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
#         edge_attr = None

#     elif dataset_name == 'METR-LA':
#         # Load speed data
#         with h5py.File('dataset/metr-la.h5', 'r') as f:
#             speed_data = pd.DataFrame(f['df']['block0_values'][:])

#         # Load adjacency matrix
#         with open('dataset/adj_mx.pkl', 'rb') as f:
#             adj_mx = pickle.load(f,encoding='latin1')
#         adjacency_matrix = adj_mx[2]

#     # Common preprocessing
#     scaler = StandardScaler()
#     speed_data = scaler.fit_transform(speed_data)

#     # Create edge_index and edge_attr
#     if dataset_name == 'PeMSD7':
#         edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
#         edge_attr = None
#     else:
#         adj_sparse = sp.coo_matrix(adjacency_matrix)
#         edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
#         edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

#     return speed_data, edge_index, edge_attr

# # --------------------------
# # 2. Model Definition
# # --------------------------
# class MLP(nn.Module):
#     def __init__(self, in_channels, hidden_layers, out_channels):
#         super().__init__()
#         layers = []
#         prev = in_channels
#         for h in hidden_layers:
#             layers.extend([nn.Linear(prev, h), nn.ReLU()])
#             prev = h
#         layers.append(nn.Linear(prev, out_channels))
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.mlp(x)

# class GRUDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_gru=64):
#         super().__init__()
#         self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
#         self.mlp = MLP(hidden_gru, [128, 64, 32], out_channels)

#     def forward(self, z):
#         gru_out, _ = self.gru(z)
#         return self.mlp(gru_out.squeeze(1))

# class GNNModel(nn.Module):
#     def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None, encoder_type='Transformer'):
#         super().__init__()
#         self.edge_index = edge_index
#         self.edge_attr = edge_attr

#         # Encoder selection
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
#             edge_dim = edge_attr.size(1) if edge_attr is not None else None
#             self.encoder = nn.Sequential(
#                 TransformerConv(input_size, hidden_channels, edge_dim=edge_dim),
#                 nn.ReLU(),
#                 TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
#             )
#         else:
#             raise ValueError(f"Unknown encoder type: {encoder_type}")

#         self.decoder = GRUDecoder(hidden_channels, out_channels)

#     def forward(self, x):
#         if isinstance(self.encoder[0], TransformerConv):
#             encoded = self.encoder[0](x, self.edge_index, self.edge_attr)
#             encoded = self.encoder[1](encoded)
#             encoded = self.encoder[2](encoded, self.edge_index, self.edge_attr)
#         else:
#             encoded = self.encoder[0](x, self.edge_index)
#             encoded = self.encoder[1](encoded)
#             encoded = self.encoder[2](encoded, self.edge_index)
#         return self.decoder(encoded.unsqueeze(1))

# # --------------------------
# # 3. Training & Evaluation
# # --------------------------
# def train_evaluate(dataset_name, encoder_type='Transformer', epochs=50, lr=0.001):
#     # Load data
#     speed_data, edge_index, edge_attr = load_dataset(dataset_name)
#     input_size = speed_data.shape[1]

#     # Convert to tensors
#     tensor_data = torch.tensor(speed_data, dtype=torch.float32)
#     train_size = int(len(tensor_data) * 0.8)
#     train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]
#     print(f"\n\nInput data shape: {train_data[:-1].shape}")  # Should be (num_time_steps - 1, num_sensors)
#     print(f"\n\nTarget data shape: {train_data[1:].shape}")  # Should be (num_time_steps - 1, num_sensors)
#     # Create model
#     model = GNNModel(
#         input_size=input_size,
#         hidden_channels=64,
#         out_channels=input_size,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         encoder_type=encoder_type
#     )
#     optimizer = Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     # Training
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         output = model(train_data[:-1])
#         loss = loss_fn(output, train_data[1:])
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#     # Evaluation
#     model.eval()
#     with torch.no_grad():
#         output = model(test_data[:-1])
#         mse = mean_squared_error(test_data[1:].numpy(), output.numpy())
#         mae = mean_absolute_error(test_data[1:].numpy(), output.numpy())
#         rmse = np.sqrt(mse)
#         mape = np.mean(np.abs((test_data[1:].numpy() - output.numpy()) / test_data[1:].numpy()))
#         print(f'\n\nMSE: {mse}')
#         print(f'MAE: {mae}')
#         print(f'MAPE: {mape}')
#         print(f'RMSE: {rmse}')


#     return model

# # --------------------------
# # 4. Run Experiments
# # --------------------------
# if __name__ == "__main__":
#     # For PeMSD7
#     train_evaluate('PeMSD7', encoder_type='GAT')
#     train_evaluate('PeMSD7', encoder_type='SAGE')
#     train_evaluate('PeMSD7', encoder_type='General')
#     train_evaluate('PeMSD7', encoder_type='Transformer')

#     # For METR-LA
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
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv
import h5py
import pickle
import scipy.sparse as sp

# --------------------------
# 1. Dataset Handling
# --------------------------
def load_dataset(dataset_name):
    if dataset_name == 'PeMSD7':
        # Load PeMSD7 data
        speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
        adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
        edge_attr = None

    elif dataset_name == 'METR-LA':
        # Load METR-LA data
        with h5py.File('dataset/metr-la.h5', 'r') as f:
            speed_data = pd.DataFrame(f['df']['block0_values'][:])

        # Load adjacency matrix
        with open('dataset/adj_mx.pkl', 'rb') as f:
            adj_mx = pickle.load(f, encoding='latin1')
        adjacency_matrix = adj_mx[2]

    elif dataset_name == 'Chickenpox':
        # Load Chickenpox data
        speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
        adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
        edge_attr = None

        # Remove non-numeric columns (e.g., dates)
        speed_data = speed_data.select_dtypes(include=[np.number])

    # Common preprocessing
    scaler = StandardScaler()
    speed_data = scaler.fit_transform(speed_data)

    # Create edge_index and edge_attr
    if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
        edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
        edge_attr = None
    else:
        adj_sparse = sp.coo_matrix(adjacency_matrix)
        edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
        edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

    return speed_data, edge_index, edge_attr

# --------------------------
# 2. Model Definition
# --------------------------
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels):
        super().__init__()
        layers = []
        prev = in_channels
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class GRUDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_gru=64):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
        self.mlp = MLP(hidden_gru, [128, 64, 32], out_channels)

    def forward(self, z):
        gru_out, _ = self.gru(z)
        return self.mlp(gru_out.squeeze(1))

class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None, encoder_type='Transformer'):
        super().__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # Encoder selection
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
            edge_dim = edge_attr.size(1) if edge_attr is not None else None
            self.encoder = nn.Sequential(
                TransformerConv(input_size, hidden_channels, edge_dim=edge_dim),
                nn.ReLU(),
                TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.decoder = GRUDecoder(hidden_channels, out_channels)

    def forward(self, x):
        if isinstance(self.encoder[0], TransformerConv):
            encoded = self.encoder[0](x, self.edge_index, self.edge_attr)
            encoded = self.encoder[1](encoded)
            encoded = self.encoder[2](encoded, self.edge_index, self.edge_attr)
        else:
            encoded = self.encoder[0](x, self.edge_index)
            encoded = self.encoder[1](encoded)
            encoded = self.encoder[2](encoded, self.edge_index)
        return self.decoder(encoded.unsqueeze(1))

# --------------------------
# 3. Training & Evaluation
# --------------------------
def train_evaluate(dataset_name, encoder_type='Transformer', epochs=50, lr=0.001):
    # Load data
    speed_data, edge_index, edge_attr = load_dataset(dataset_name)
    input_size = speed_data.shape[1]

    # Convert to tensors
    tensor_data = torch.tensor(speed_data, dtype=torch.float32)
    train_size = int(len(tensor_data) * 0.8)
    train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]

    # Create model
    model = GNNModel(
        input_size=input_size,
        hidden_channels=64,
        out_channels=input_size,
        edge_index=edge_index,
        edge_attr=edge_attr,
        encoder_type=encoder_type
    )
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data[:-1])
        loss = loss_fn(output, train_data[1:])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print("\n\n")

    # Evaluation
    model.eval()
    with torch.no_grad():
        output = model(test_data[:-1])
        mse = mean_squared_error(test_data[1:].numpy(), output.numpy())
        mae = mean_absolute_error(test_data[1:].numpy(), output.numpy())
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_data[1:].numpy() - output.numpy()) / test_data[1:].numpy()))
        print(f'MSE: {mse}')
        print(f'MAE: {mae}')
        print(f'MAPE: {mape}')
        print(f'RMSE: {rmse}')
    print("\n\n")

    return model

# --------------------------
# 4. Run Experiments
# --------------------------
if __name__ == "__main__":
    # For PeMSD7

    print ("PeMSD7")
    print ("--"*20)
    print("Encoder Type: GAT")
    train_evaluate('PeMSD7', encoder_type='GAT')
    print ("--"*20)
    print("Encoder Type: SAGE")
    train_evaluate('PeMSD7', encoder_type='SAGE')
    print ("--"*20)
    print("Encoder Type: General")
    train_evaluate('PeMSD7', encoder_type='General')
    print ("--"*20)
    print("Encoder Type: Transformer")
    train_evaluate('PeMSD7', encoder_type='Transformer')
    print ("--"*20)
    print ("--"*20)
    print()
    print()
    print()


    # For METR-LA
    print ("METR-LA")
    print ("--"*20)
    print("Encoder Type: GAT")
    train_evaluate('METR-LA', encoder_type='GAT')
    print ("--"*20)
    print("Encoder Type: SAGE")
    train_evaluate('METR-LA', encoder_type='SAGE')
    print ("--"*20)
    print("Encoder Type: General")
    train_evaluate('METR-LA', encoder_type='General')
    print ("--"*20)
    print("Encoder Type: Transformer")
    train_evaluate('METR-LA', encoder_type='Transformer')
    print ("--"*20)
    print ("--"*20)
    print()
    print()
    print()


    # For Chickenpox
    print ("Chickenpox")
    print ("--"*20)
    print("Encoder Type: GAT")
    train_evaluate('Chickenpox', encoder_type='GAT')
    print ("--"*20)
    print("Encoder Type: SAGE")
    train_evaluate('Chickenpox', encoder_type='SAGE')
    print ("--"*20)
    print("Encoder Type: General")
    train_evaluate('Chickenpox', encoder_type='General')
    print ("--"*20)
    print("Encoder Type: Transformer")
    train_evaluate('Chickenpox', encoder_type='Transformer')


