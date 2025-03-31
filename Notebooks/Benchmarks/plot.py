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
import plotly.graph_objects as go
import plotly.io as pio

# Set the default renderer to avoid using MathJax
pio.renderers.default = 'svg'
# --------------------------
# 1. Dataset Handling
# --------------------------
def load_dataset(dataset_name):
    if dataset_name == 'PeMSD7':
        speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
        adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
        edge_attr = None
    elif dataset_name == 'METR-LA':
        with h5py.File('dataset/metr-la.h5', 'r') as f:
            speed_data = pd.DataFrame(f['df']['block0_values'][:])
        with open('dataset/adj_mx.pkl', 'rb') as f:
            adj_mx = pickle.load(f, encoding='latin1')
        adjacency_matrix = adj_mx[2]
    elif dataset_name == 'Chickenpox':
        speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
        adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
        edge_attr = None
        speed_data = speed_data.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    speed_data = scaler.fit_transform(speed_data)

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

    # Training with loss collection
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data[:-1])
        loss = loss_fn(output, train_data[1:])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Plot and save loss curve for TransformerConv
    if encoder_type == 'Transformer':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(epochs)), y=losses, mode='lines', name='Loss'))
        fig.update_layout(
            # title=f'Loss Curve for {dataset_name} with TransformerConv',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            plot_bgcolor='white',  # Set background color to white
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        # Save as PDF
        fig.write_image(f"loss_curve_{dataset_name}_transformer.svg", engine="kaleido")
        import cairosvg
        cairosvg.svg2pdf(url=f"loss_curve_{dataset_name}_transformer.svg",
                 write_to=f"loss_curve_{dataset_name}_transformer.pdf")

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

    # Plot and save actual vs predicted for a random node for TransformerConv
    if encoder_type == 'Transformer':
        num_nodes = test_data.shape[1]
        node_idx = num_nodes//2
        actual = test_data[1:, node_idx].numpy()
        predicted = output[:, node_idx].numpy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(actual))), y=actual, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=list(range(len(predicted))), y=predicted, mode='lines', name='Predicted'))
        fig.update_layout(
            # title=f'Actual vs Predicted for Node {node_idx} in {dataset_name}',
            xaxis_title='Time Step',
            yaxis_title='Value (Scaled)',
            plot_bgcolor='white',  # Set background color to white
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        # Save as PDF
        fig.write_image(f"actual_vs_predicted_{dataset_name}_node_{node_idx}.svg",engine="kaleido")
        import cairosvg
        cairosvg.svg2pdf(url=f"actual_vs_predicted_{dataset_name}_node_{node_idx}.svg",
                 write_to=f"actual_vs_predicted_{dataset_name}_node_{node_idx}.pdf")

    print("\n\n")
    return model

# --------------------------
# 4. Run Experiments
# --------------------------
if __name__ == "__main__":
    datasets = ['PeMSD7']
    encoder_types = ['Transformer']

    for dataset in datasets:
        print(dataset)
        print("--" * 20)
        for encoder in encoder_types:
            print(f"Encoder Type: {encoder}")
            train_evaluate(dataset, encoder_type=encoder)
            print("--" * 20)
        print("--" * 20)
        print("\n\n")
