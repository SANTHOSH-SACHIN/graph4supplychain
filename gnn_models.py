import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
from torch_geometric.nn import global_mean_pool

warnings.filterwarnings("ignore")
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env.default')

# GNN models
import torch
from torch_geometric.nn import to_hetero
from torch_geometric.nn.conv import GATConv, SAGEConv

import torch.nn.functional as F

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from torch_geometric.nn import SAGEConv, GATConv, GeneralConv , TransformerConv, to_hetero
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels):
        super().__init__()
        layers = []
        prev_layer = in_channels

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_layer, hidden_dim))
            layers.append(nn.ReLU())
            prev_layer = hidden_dim

        layers.append(nn.Linear(prev_layer, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GNNEncoder2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x


class GNNEncoder3(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels, in_edge_channels=-1)
        self.conv2 = GeneralConv((-1, -1), out_channels, in_edge_channels=-1)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x

class GNNEncoder4(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0, edge_dim=None):
        super().__init__()
        
        # First TransformerConv layer
        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            beta=False,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True
        )
        
        # Second TransformerConv layer
        self.conv2 = TransformerConv(
            in_channels=hidden_channels * heads,  # Since concat=True, we multiply by heads
            out_channels=out_channels,
            heads=heads,
            concat=True,
            beta=False,
            dropout=dropout,
            edge_dim=edge_dim,
            bias=True,
            root_weight=True
        )

    def forward(self, x, edge_index, edge_attr=None):
        # First layer
        x = self.conv1(x, edge_index, edge_attr).relu()
        
        # Second layer
        x = self.conv2(x, edge_index, edge_attr)
        
        return x



class GRUDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_num_parts, hidden_gru=64):
        super().__init__()
        hidden_layers = [128, 64, 32]

        self.part_grus = nn.ModuleList(
            [
                nn.GRU(in_channels, hidden_gru, batch_first=True)
                for _ in range(max_num_parts)
            ]
        )

        self.part_mlps = nn.ModuleList(
            [MLP(hidden_gru, hidden_layers, out_channels) for _ in range(max_num_parts)]
        )


        self.in_channels = in_channels
        self.hidden_gru = hidden_gru

    def forward(self, z_dict, num_parts, label):
        z = z_dict[label]
        outputs = []

        for part_idx in range(num_parts):
            part_embedding = z[part_idx].unsqueeze(0).unsqueeze(0)
            gru_out, _ = self.part_grus[part_idx](part_embedding)
            part_output = self.part_mlps[part_idx](gru_out.squeeze())
            outputs.append(part_output)

        return torch.stack(outputs)


class MultistepDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_steps, max_number_parts, hidden_gru=64):
        super().__init__()

        # Define the MLP layers
        hidden_layers = [128, 64, 32]

        self.part_grus = nn.ModuleList(
            [
                nn.GRU(in_channels, hidden_gru, batch_first=True)
                for _ in range(max_number_parts)
            ]
        )
        self.part_mlps = nn.ModuleList(
            [MLP(hidden_gru, hidden_layers, out_channels*out_steps) for _ in range(max_number_parts)]
        )
        self.out_steps = out_steps
        self.out_channels = out_channels

    def forward(self, z_dict):
        z = z_dict["PARTS"]
        outputs=[]
        for part_idx in range(len(z)):
            part_embedding = z[part_idx]
            part_gru, _ = self.part_grus[part_idx](part_embedding.unsqueeze(0))
            part_output = self.part_mlps[part_idx](part_gru.squeeze())
            outputs.append(part_output)
        outputs = torch.stack(outputs)
        if self.out_channels>1:
                outputs = outputs.view(len(z), self.out_channels, self.out_steps)
        return outputs

class MultiStepModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, out_steps, max_number_parts, G=None):
        super().__init__()
        # Dynamically choose encoder based on user configuration
        self.encoder_type = st.session_state.get("encoder_type", "SAGEConv")

        if self.encoder_type == "SAGEConv":
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        elif self.encoder_type == "GATConv":
            self.encoder = GNNEncoder2(hidden_channels, hidden_channels)
        elif self.encoder_type == "GeneralConv":
            self.encoder = GNNEncoder3(hidden_channels, hidden_channels)
        else:
            self.encoder = GNNEncoder4(hidden_channels, hidden_channels)

        if G is not None:
            self.encoder = to_hetero(self.encoder, G.metadata(), aggr="mean")

        self.decoder = MultistepDecoder(
            hidden_channels, out_channels, out_steps=out_steps, max_number_parts=max_number_parts
        )

    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        if self.encoder_type == "SAGEConv":
            z_dict = self.encoder(x_dict, edge_index_dict)
        else:
            z_dict = self.encoder(x_dict, edge_index_dict, edge_attr)
        return self.decoder(z_dict)


def train_multistep_regression(
    num_epochs,
    model,
    optimizer,
    loss_fn,
    temporal_graphs,
    label="PARTS",
    device="cpu",
    patience=5,
):
    st.subheader("Training Progress . . .")
    progress_bar = st.progress(0)
    status_text = st.empty()

    best_test_r2 = -float("inf")
    best_test_adjusted_r2 = -float("inf")
    best_test_mae = float("inf")
    patience_counter = 0
    best_state = None
    epoch_losses = []
    epoch_r2_scores = []
    epoch_adjusted_r2_scores = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0
        train_true_all = []
        train_pred_all = []
        epoch_test_loss = 0.0
        epoch_test_mae = 0.0
        test_true_all = []
        test_pred_all = []
        graph_count = 0

        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            graph_count += 1

            # Training phase
            model.train()
            optimizer.zero_grad()

            # Forward pass
            out = model(G.x_dict, G.edge_index_dict).squeeze(1)

            # Get training masks and compute loss
            train_mask = G[label]["train_mask"]
            train_loss = loss_fn(out[train_mask], G[label].y[train_mask])

            # Backward pass
            train_loss.backward()
            optimizer.step()

            # Compute training metrics
            with torch.no_grad():
                train_pred = out[train_mask].cpu().numpy()
                train_true = G[label].y[train_mask].cpu().numpy()
                train_mae = (
                    torch.abs(out[train_mask] - G[label].y[train_mask]).mean().item()
                )

                # Collect all predictions and true values
                train_pred_all.append(train_pred)
                train_true_all.append(train_true)
                epoch_train_loss += train_loss.item()
                epoch_train_mae += train_mae

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                test_mask = G[label]["test_mask"]
                test_pred = out[test_mask].cpu().numpy()
                test_true = G[label].y[test_mask].cpu().numpy()

                test_loss = loss_fn(out[test_mask], G[label].y[test_mask])
                test_mae = (
                    torch.abs(out[test_mask] - G[label].y[test_mask]).mean().item()
                )

                # Collect all predictions and true values
                test_pred_all.append(test_pred)
                test_true_all.append(test_true)
                epoch_test_loss += test_loss.item()
                epoch_test_mae += test_mae

        # Aggregate all predictions and true values
        train_true_all = np.concatenate(train_true_all)
        train_pred_all = np.concatenate(train_pred_all)
        test_true_all = np.concatenate(test_true_all)
        test_pred_all = np.concatenate(test_pred_all)

        # Compute R² and Adjusted R² score over all data
        def compute_adjusted_r2(y_true, y_pred, n_features):
            n_samples = len(y_true)
            r2 = r2_score(y_true, y_pred)
            adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
            return r2, adjusted_r2

        # Assuming n_features is the number of input features (you may need to adjust this)
        n_features = train_pred_all.shape[1] if train_pred_all.ndim > 1 else 1

        avg_train_r2, avg_train_adjusted_r2 = compute_adjusted_r2(train_true_all, train_pred_all, n_features)
        avg_test_r2, avg_test_adjusted_r2 = compute_adjusted_r2(test_true_all, test_pred_all, n_features)

        # Average losses and MAEs
        avg_train_loss = epoch_train_loss / graph_count
        avg_train_mae = epoch_train_mae / graph_count
        avg_test_loss = epoch_test_loss / graph_count
        avg_test_mae = epoch_test_mae / graph_count

        epoch_losses.append(avg_test_loss)
        epoch_r2_scores.append(avg_test_r2)
        epoch_adjusted_r2_scores.append(avg_test_adjusted_r2)

        # Check for improvement based on adjusted R²
        if avg_test_adjusted_r2 > best_test_adjusted_r2:
            best_test_mae = avg_test_loss
            best_test_r2 = avg_test_r2
            best_test_adjusted_r2 = avg_test_adjusted_r2
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            st.write(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Update progress bar
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train R²: {avg_test_r2:.4f}, Train Adjusted R²: {avg_test_adjusted_r2:.4f}"
        )

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    return model, best_test_r2, best_test_adjusted_r2, best_test_mae, epoch_losses, epoch_r2_scores, epoch_adjusted_r2_scores


def train_multistep_classification(
    num_epochs,
    model,
    optimizer,
    loss_fn,
    temporal_graphs,
    label="PARTS",
    device="cpu",
    patience=5,
):
    st.subheader("Multistep Classification Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    best_test_accuracy = 0.0
    best_test_loss = float("inf")
    patience_counter = 0
    best_state = None
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_test_loss = 0.0
        epoch_test_accuracy = 0.0
        graph_count = 0

        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            graph_count += 1

            # Training phase
            model.train()
            optimizer.zero_grad()

            # Forward pass
            out = model(G.x_dict, G.edge_index_dict, G.edge_attr)

            # Get training masks and compute loss
            # st.write(out.shape)
            train_mask = G[label]["train_mask"]
            train_loss = loss_fn(out[train_mask], G[label].y[train_mask])

            # Backward pass
            train_loss.backward()
            optimizer.step()

            # Compute training metrics
            with torch.no_grad():
                train_pred = out[train_mask].argmax(dim=1).cpu()
                train_true = G[label].y[train_mask].cpu()
                correct = (train_pred == train_true).all(dim=1)
                train_accuracy = correct.sum().item() / len(correct)

            # Update epoch metrics for training
            epoch_train_loss += train_loss.item()
            epoch_train_accuracy += train_accuracy

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                # Forward pass for testing
                test_mask = G[label]["test_mask"]
                test_pred = out[test_mask].argmax(dim=1).cpu()
                test_true = G[label].y[test_mask].cpu()

                # Compute test metrics
                test_loss = loss_fn(out[test_mask], G[label].y[test_mask])
                correct = (test_pred == test_true).all(dim=1)
                test_accuracy = correct.sum().item() / len(correct)

                # Update epoch metrics for testing
                epoch_test_loss += test_loss.item()
                epoch_test_accuracy += test_accuracy

        # Average metrics over all graphs
        avg_train_loss = epoch_train_loss / graph_count
        avg_train_accuracy = epoch_train_accuracy / graph_count
        avg_test_loss = epoch_test_loss / graph_count
        avg_test_accuracy = epoch_test_accuracy / graph_count

        # Store epoch metrics
        epoch_losses.append(avg_test_loss)
        epoch_accuracies.append(avg_test_accuracy)

        # Check for improvement
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_test_accuracy = avg_test_accuracy
            best_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            st.write(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Update progress bar and status
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}"
        )

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model, best_test_accuracy, best_test_loss, epoch_losses, epoch_accuracies

class Model3(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, max_num_parts, G=None):
        super().__init__()
        # Dynamically choose encoder based on user configuration
        self.encoder_type = st.session_state.get("encoder_type", "SAGEConv")

        if self.encoder_type == "SAGEConv":
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        elif self.encoder_type == "GATConv":
            self.encoder = GNNEncoder2(hidden_channels, hidden_channels)
        elif self.encoder_type == "GeneralConv":
            self.encoder = GNNEncoder3(hidden_channels, hidden_channels)
        else:
            self.encoder = GNNEncoder4(hidden_channels, hidden_channels)

        if G is not None:
            self.encoder = to_hetero(self.encoder, G.metadata(), aggr="mean")

        self.decoder = GRUDecoder(hidden_channels, out_channels, max_num_parts)

    def forward(self, x_dict, edge_index_dict, num_parts, label, edge_attr=None):
        if self.encoder_type == "SAGEConv":
            z_dict = self.encoder(x_dict, edge_index_dict)
        else:
            z_dict = self.encoder(x_dict, edge_index_dict, edge_attr)
        return self.decoder(z_dict, num_parts, label)
    def __getstate__(self):
        # Return a picklable state
        state = self.__dict__.copy()
        # Remove unpicklable entries if needed
        # del state['some_unpicklable_attribute']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)


def train_classification(
    num_epochs,
    model,
    optimizer,
    loss_fn,
    temporal_graphs,
    label="PARTS",
    device="cpu",
    patience=5,
):
    st.subheader("Training Progress . . .")
    progress_bar = st.progress(0)
    status_text = st.empty()
    best_test_accuracy = 0.0
    best_test_loss = float("inf")
    patience_counter = 0
    best_state = None
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_test_loss = 0.0
        epoch_test_accuracy = 0.0
        graph_count = 0

        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            graph_count += 1

            # Training phase
            model.train()
            optimizer.zero_grad()

            # Forward pass
            out = model(G.x_dict, G.edge_index_dict,len(G[label].y),label, G.edge_attr)
            print(out.shape)
            # Get training masks and compute loss
            train_mask = G[label]["train_mask"]
            train_loss = loss_fn(out[train_mask], G[label].y[train_mask])

            # Backward pass
            train_loss.backward()
            optimizer.step()

            # Compute training metrics
            with torch.no_grad():
                train_pred = out[train_mask].argmax(dim=1).cpu()
                train_true = G[label].y[train_mask].cpu()
                train_accuracy = accuracy_score(train_true, train_pred)

            # Update epoch metrics for training
            epoch_train_loss += train_loss.item()
            epoch_train_accuracy += train_accuracy

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                # Forward pass for testing
                test_mask = G[label]["test_mask"]
                test_pred = out[test_mask].argmax(dim=1).cpu()
                test_true = G[label].y[test_mask].cpu()

                # Compute test metrics
                test_loss = loss_fn(out[test_mask], G[label].y[test_mask])
                test_accuracy = accuracy_score(test_true, test_pred)

                # Update epoch metrics for testing
                epoch_test_loss += test_loss.item()
                epoch_test_accuracy += test_accuracy

        # Average metrics over all graphs
        avg_train_loss = epoch_train_loss / graph_count
        avg_train_accuracy = epoch_train_accuracy / graph_count
        avg_test_loss = epoch_test_loss / graph_count
        avg_test_accuracy = epoch_test_accuracy / graph_count

        epoch_losses.append(avg_test_loss)
        epoch_accuracies.append(avg_test_accuracy)

        # Check for improvement
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_test_accuracy = avg_test_accuracy
            best_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            st.write(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Update progress bar
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}"
        )

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    return model, best_test_accuracy, best_test_loss, epoch_losses, epoch_accuracies


# def train_regression(
#     num_epochs,
#     model,
#     optimizer,
#     loss_fn,
#     temporal_graphs,
#     label="PARTS",
#     device="cpu",
#     patience=5,
# ):
#     st.subheader("Training Progress . . .")
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     best_test_r2 = -float("inf")
#     best_test_mae = float("inf")
#     patience_counter = 0
#     best_state = None
#     epoch_losses = []
#     epoch_r2_scores = []

#     for epoch in range(num_epochs):
#         epoch_train_loss = 0.0
#         epoch_test_loss = 0.0
#         train_mse = 0.0
#         train_mae = 0.0
#         test_mse = 0.0
#         test_mae = 0.0
#         test_r2 = 0.0
#         graph_count = 0

#         for row in temporal_graphs:
#             G = temporal_graphs[row][1]
#             graph_count += 1

#             # Training phase
#             model.train()
#             optimizer.zero_grad()

#             # Forward pass
#             out = model(G.x_dict, G.edge_index_dict, len(G[label].y),label, G.edge_attr).squeeze(-1)

#             # Get training masks and compute loss
#             train_mask = G[label]["train_mask"]
#             train_loss = loss_fn(out[train_mask], G[label].y[train_mask])

#             # Backward pass
#             train_loss.backward()
#             optimizer.step()

#             # Compute training metrics
#             with torch.no_grad():
#                 train_pred = out[train_mask].cpu().numpy()
#                 train_true = G[label].y[train_mask].cpu().numpy()
#                 train_mse += mean_squared_error(train_true, train_pred)
#                 train_mae += mean_absolute_error(train_true, train_pred)

#             # Update epoch metrics for training
#             epoch_train_loss += train_loss.item()

#             # Evaluation phase
#             model.eval()
#             with torch.no_grad():
#                 # Forward pass for testing
#                 test_mask = G[label]["test_mask"]
#                 test_pred = out[test_mask].cpu().numpy()
#                 test_true = G[label].y[test_mask].cpu().numpy()

#                 # Compute test metrics
#                 test_loss = loss_fn(out[test_mask], G[label].y[test_mask])
#                 test_mse += mean_squared_error(test_true, test_pred)
#                 test_mae += mean_absolute_error(test_true, test_pred)
#                 test_r2 += r2_score(test_true, test_pred)

#                 # Update epoch metrics for testing
#                 epoch_test_loss += test_loss.item()

#         # Average metrics over all graphs
#         avg_train_loss = epoch_train_loss / graph_count
#         avg_test_loss = epoch_test_loss / graph_count
#         avg_train_mse = train_mse / graph_count
#         avg_train_mae = train_mae / graph_count
#         avg_test_mse = test_mse / graph_count
#         avg_test_mae = test_mae / graph_count
#         avg_test_r2 = test_r2 / graph_count

#         epoch_losses.append(avg_test_loss)
#         epoch_r2_scores.append(avg_test_r2)

#         # Check for improvement
#         if avg_test_loss < best_test_mae:
#             best_test_mae = avg_test_loss
#             best_test_r2 = avg_test_r2
#             best_state = model.state_dict().copy()
#             patience_counter = 0  # Reset patience counter
#         else:
#             patience_counter += 1

#         # Early stopping
#         if patience_counter >= patience:
#             st.write(f"Early stopping triggered at epoch {epoch + 1}")
#             break

#         # Update progress bar
#         progress_bar.progress((epoch + 1) / num_epochs)
#         status_text.text(
#             f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss**0.5:.4f}, Train R²: {avg_test_r2:.4f}"
#         )

#     # Load best model
#     if best_state:
#         model.load_state_dict(best_state)
#     return model, best_test_r2, best_test_mae**0.5, epoch_losses, epoch_r2_scores


# def test_single_step_regression(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
#     model.eval()
#     test_true_all = []
#     test_pred_all = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for row in temporal_graphs:
#             G = temporal_graphs[row][1]
#             test_mask = G[label]
#             out = model(G.x_dict, G.edge_index_dict, len(G[label].y),label, G.edge_attr).squeeze(-1)
#             test_loss = loss_fn(out, G[label].y)
#             total_loss += test_loss.item()

#             test_pred = out.cpu().numpy()
#             test_true = G[label].y.cpu().numpy()
#             test_pred_all.append(test_pred)
#             test_true_all.append(test_true)

#     test_true_all = np.concatenate(test_true_all)
#     test_pred_all = np.concatenate(test_pred_all)
#     r2 = r2_score(test_true_all, test_pred_all)
#     mae = mean_absolute_error(test_true_all, test_pred_all)

#     return {"R2": r2, "MAE": mae, "Loss": total_loss / len(temporal_graphs), "Predictions":test_pred_all}


def train_regression(
    num_epochs,
    model,
    optimizer,
    loss_fn,
    temporal_graphs,
    label="PARTS",
    device="cpu",
    patience=5,
):
    st.subheader("Training Progress . . .")
    progress_bar = st.progress(0)
    status_text = st.empty()

    best_test_r2 = -float("inf")
    best_test_adjusted_r2 = -float("inf")
    best_test_mae = float("inf")
    patience_counter = 0
    best_state = None
    epoch_losses = []
    epoch_r2_scores = []
    epoch_adjusted_r2_scores = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        test_mse = 0.0
        test_mae = 0.0
        test_r2 = 0.0
        test_adjusted_r2 = 0.0
        graph_count = 0

        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            graph_count += 1

            # Training phase
            model.train()
            optimizer.zero_grad()

            # Forward pass
            out = model(G.x_dict, G.edge_index_dict, len(G[label].y), label, G.edge_attr).squeeze(-1)

            # Get training masks and compute loss
            train_mask = G[label]["train_mask"]
            train_loss = loss_fn(out[train_mask], G[label].y[train_mask])

            # Backward pass
            train_loss.backward()
            optimizer.step()

            # Compute training metrics
            with torch.no_grad():
                train_pred = out[train_mask].cpu().numpy()
                train_true = G[label].y[train_mask].cpu().numpy()
                train_mse += mean_squared_error(train_true, train_pred)
                train_mae += mean_absolute_error(train_true, train_pred)

            # Update epoch metrics for training
            epoch_train_loss += train_loss.item()

            # Evaluation phase
            model.eval()
            with torch.no_grad():
                # Forward pass for testing
                test_mask = G[label]["test_mask"]
                test_pred = out[test_mask].cpu().numpy()
                test_true = G[label].y[test_mask].cpu().numpy()

                # Compute test metrics
                test_loss = loss_fn(out[test_mask], G[label].y[test_mask])
                test_mse += mean_squared_error(test_true, test_pred)
                test_mae += mean_absolute_error(test_true, test_pred)
                
                # Calculate R² and Adjusted R²
                n = len(test_true)  # number of observations
                p = out.shape[1] if len(out.shape) > 1 else 1  # number of predictors
                test_r2_value = r2_score(test_true, test_pred)
                test_adjusted_r2_value = 1 - (1 - test_r2_value) * (n - 1) / (n - p - 1)
                
                test_r2 += test_r2_value
                test_adjusted_r2 += test_adjusted_r2_value

                # Update epoch metrics for testing
                epoch_test_loss += test_loss.item()

        # Average metrics over all graphs
        avg_train_loss = epoch_train_loss / graph_count
        avg_test_loss = epoch_test_loss / graph_count
        avg_train_mse = train_mse / graph_count
        avg_train_mae = train_mae / graph_count
        avg_test_mse = test_mse / graph_count
        avg_test_mae = test_mae / graph_count
        avg_test_r2 = test_r2 / graph_count
        avg_test_adjusted_r2 = test_adjusted_r2 / graph_count

        epoch_losses.append(avg_test_loss)
        epoch_r2_scores.append(avg_test_r2)
        epoch_adjusted_r2_scores.append(avg_test_adjusted_r2)

        # Check for improvement based on adjusted R²
        if avg_test_adjusted_r2 > best_test_adjusted_r2:
            best_test_mae = avg_test_loss
            best_test_r2 = avg_test_r2
            best_test_adjusted_r2 = avg_test_adjusted_r2
            best_state = model.state_dict().copy()
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            st.write(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Update progress bar
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss**0.5:.4f}, Train R²: {avg_test_r2:.4f}, Adjusted R²: {avg_test_adjusted_r2:.4f}"
        )

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_test_r2, best_test_adjusted_r2, best_test_mae**0.5, epoch_losses, epoch_r2_scores, epoch_adjusted_r2_scores

def test_single_step_regression(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
    model.eval()
    test_true_all = []
    test_pred_all = []
    total_loss = 0.0

    with torch.no_grad():
        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            test_mask = G[label]
            out = model(G.x_dict, G.edge_index_dict, len(G[label].y),label, G.edge_attr).squeeze(-1)
            test_loss = loss_fn(out, G[label].y)
            total_loss += test_loss.item()

            test_pred = out.cpu().numpy()
            test_true = G[label].y.cpu().numpy()
            test_pred_all.append(test_pred)
            test_true_all.append(test_true)

    test_true_all = np.concatenate(test_true_all)
    test_pred_all = np.concatenate(test_pred_all)
    n_samples = len(test_true_all)
    n_features = test_pred_all.shape[1] if test_pred_all.ndim > 1 else 1
    
    r2 = r2_score(test_true_all, test_pred_all)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    mae = mean_absolute_error(test_true_all, test_pred_all)

    return {"R2": r2 ,"Adjusted R2" : adjusted_r2 , "MAE": mae, "Loss": total_loss / len(temporal_graphs), "Predictions":test_pred_all}

# def test_single_step_regression(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
#     model.eval()
#     test_true_all = []
#     test_pred_all = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for row in temporal_graphs:
#             G = temporal_graphs[row][1]
#             test_mask = G[label]
#             out = model(G.x_dict, G.edge_index_dict, len(G[label].y), label, G.edge_attr).squeeze(-1)
#             test_loss = loss_fn(out, G[label].y)
#             total_loss += test_loss.item()

#             test_pred = out.cpu().numpy()
#             test_true = G[label].y.cpu().numpy()
            
#             # Calculate adjusted R²
#             n = len(test_true)  # number of observations
#             p = out.shape[1] if len(out.shape) > 1 else 1  # number of predictors
#             r2 = r2_score(test_true, test_pred)
#             adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

#             test_pred_all.append(test_pred)
#             test_true_all.append(test_true)

#     test_true_all = np.concatenate(test_true_all)
#     test_pred_all = np.concatenate(test_pred_all)
#     r2 = r2_score(test_true_all, test_pred_all)
#     adjusted_r2 = 1 - (1 - r2) * (len(test_true_all) - 1) / (len(test_true_all) - out.shape[1] - 1)
#     mae = mean_absolute_error(test_true_all, test_pred_all)

#     return {
#         "R2": r2, 
#         "Adjusted R2": adjusted_r2, 
#         "MAE": mae, 
#         "Loss": total_loss / len(temporal_graphs), 
#         "Predictions": test_pred_all
#     }



def test_single_step_classification(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
    model.eval()
    test_true_all = []
    test_pred_all = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            # test_mask = G[label]["test_mask"]
            
            out = model(G.x_dict, G.edge_index_dict, len(G[label].y),label, G.edge_attr)
            test_loss = loss_fn(out, G[label].y)
            total_loss += test_loss.item()

            test_pred = out.argmax(dim=1).cpu()
            test_true = G[label].y.cpu()
            total_correct += (test_pred == test_true).sum().item()
            total_samples += len(test_true)

    accuracy = total_correct / total_samples

    return {"Accuracy": accuracy, "Loss": total_loss / len(temporal_graphs)}


# def test_multistep_regression(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
#     model.eval()
#     test_true_all = []
#     test_pred_all = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for row in temporal_graphs:
#             G = temporal_graphs[row][1]
#             # test_mask = G[label]["test_mask"]
#             out = model(G.x_dict, G.edge_index_dict).squeeze(1)
#             test_loss = loss_fn(out, G[label].y)

#             total_loss += test_loss.item()

#             test_pred = out.cpu().numpy()
#             test_true = G[label].y.cpu().numpy()
#             test_pred_all.append(test_pred)
#             test_true_all.append(test_true)

#     test_true_all = np.concatenate(test_true_all)
#     test_pred_all = np.concatenate(test_pred_all)
#     r2 = r2_score(test_true_all, test_pred_all)
#     mae = mean_absolute_error(test_true_all, test_pred_all)

#     return {"R2": r2, "MAE": mae, "Loss": total_loss / len(temporal_graphs)}

def test_multistep_regression(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
    model.eval()
    test_true_all = []
    test_pred_all = []
    total_loss = 0.0

    with torch.no_grad():
        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            # test_mask = G[label]["test_mask"]
            out = model(G.x_dict, G.edge_index_dict).squeeze(1)
            test_loss = loss_fn(out, G[label].y)

            total_loss += test_loss.item()

            test_pred = out.cpu().numpy()
            test_true = G[label].y.cpu().numpy()
            test_pred_all.append(test_pred)
            test_true_all.append(test_true)

    test_true_all = np.concatenate(test_true_all)
    test_pred_all = np.concatenate(test_pred_all)
    
    # Compute R² and Adjusted R²
    # Estimate number of features (you might need to adjust this based on your model's input)
    n_samples = len(test_true_all)
    n_features = test_pred_all.shape[1] if test_pred_all.ndim > 1 else 1
    
    r2 = r2_score(test_true_all, test_pred_all)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    mae = mean_absolute_error(test_true_all, test_pred_all)

    return {
        "R2": r2, 
        "Adjusted R2": adjusted_r2, 
        "MAE": mae, 
        "Loss": total_loss / len(temporal_graphs)
    }
    
def test_multistep_classification(model, temporal_graphs, loss_fn, label="PARTS", device="cpu"):
    model.eval()
    test_true_all = []
    test_pred_all = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            # test_mask = G[label]["test_mask"]
            out = model(G.x_dict, G.edge_index_dict, G.edge_attr)
            test_loss = loss_fn(out, G[label].y)
            total_loss += test_loss.item()

            test_pred = out.argmax(dim=1).cpu()
            test_true = G[label].y.cpu()
            total_correct += (test_pred == test_true).sum().item()
            total_samples += len(test_true)

    accuracy = total_correct / total_samples

    return {"Accuracy": accuracy, "Loss": total_loss / len(temporal_graphs)}
