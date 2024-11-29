import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os
import requests
import warnings
warnings.filterwarnings('ignore')
import copy
from dotenv import load_dotenv
import hybrid_st as hm
load_dotenv()


# Time series models
from arima_st import SingleStepARIMA, MultiStepARIMA
from sarima_st import SingleStepSarimaAnalyzer, MultiStepSarimaAnalyzer
from cnnlstm_st import SingleStepCNNLSTMAnalyzer
from xgboost_st import XGBoostForecaster
from prophet_st import SingleStepProphet, MultiStepProphet
from parser_st import TemporalHeterogeneousGraphParser

# GNN models
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import to_hetero
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.nn.conv import GATConv, SAGEConv
import plotly.graph_objects as go
import networkx as nx
import torch.nn.functional as F
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, to_hetero
import torch.nn as nn

headers = {"accept": "application/json"}

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

class GRUDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_parts, hidden_gru=64):
        super().__init__()
        hidden_layers = [128, 64, 32]
        
        self.part_grus = nn.ModuleList([
            nn.GRU(in_channels, hidden_gru, batch_first=True) 
            for _ in range(num_parts)
        ])
        
        self.part_mlps = nn.ModuleList([
            MLP(hidden_gru, hidden_layers, out_channels) 
            for _ in range(num_parts)
        ])
        
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.hidden_gru = hidden_gru
    
    def forward(self, z_dict):
        z = z_dict['PARTS']
        outputs = []
        
        for part_idx in range(self.num_parts):
            part_embedding = z[part_idx].unsqueeze(0).unsqueeze(0)
            gru_out, _ = self.part_grus[part_idx](part_embedding)
            part_output = self.part_mlps[part_idx](gru_out.squeeze())
            outputs.append(part_output)
        
        return torch.stack(outputs)

class Model3(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_parts, G=None):
        super().__init__()
        # Dynamically choose encoder based on user configuration
        self.encoder_type = st.session_state.get('encoder_type', 'SAGEConv')
        
        if self.encoder_type == 'SAGEConv':
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        elif self.encoder_type == 'GATConv':
            self.encoder = GNNEncoder2(hidden_channels, hidden_channels)
        else:
            self.encoder = GNNEncoder3(hidden_channels, hidden_channels)
        
        if G is not None:
            self.encoder = to_hetero(self.encoder, G.metadata(), aggr='mean')
        
        self.decoder = GRUDecoder(hidden_channels, out_channels, num_parts)
    
    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        if self.encoder_type == 'SAGEConv':
            z_dict = self.encoder(x_dict, edge_index_dict)
        else:
            z_dict = self.encoder(x_dict, edge_index_dict, edge_attr)
        return self.decoder(z_dict)

def train_classification(num_epochs, model, optimizer, loss_fn, temporal_graphs, label='PARTS', device='cpu', patience=5):
    st.subheader("Training Progress . . .")
    progress_bar = st.progress(0)
    status_text = st.empty()
    best_test_accuracy = 0.0
    best_test_loss = float('inf')
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
            train_mask = G[label]['train_mask']
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
                test_mask = G[label]['test_mask']
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
        status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    return model, best_test_accuracy, best_test_loss, epoch_losses, epoch_accuracies

def train_regression(num_epochs, model, optimizer, loss_fn, temporal_graphs, label='PARTS', device='cpu', patience=5):
    st.subheader("Training Progress . . .")
    progress_bar = st.progress(0)
    status_text = st.empty()

    best_test_r2 = -float('inf')
    best_test_mae = float('inf')
    patience_counter = 0
    best_state = None
    epoch_losses = []
    epoch_r2_scores = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        test_mse = 0.0
        test_mae = 0.0
        test_r2 = 0.0
        graph_count = 0

        for row in temporal_graphs:
            G = temporal_graphs[row][1]
            graph_count += 1

            # Training phase
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(G.x_dict, G.edge_index_dict, G.edge_attr).squeeze(-1)
            
            # Get training masks and compute loss
            train_mask = G[label]['train_mask']
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
                test_mask = G[label]['test_mask']
                test_pred = out[test_mask].cpu().numpy()
                test_true = G[label].y[test_mask].cpu().numpy()
                
                # Compute test metrics
                test_loss = loss_fn(out[test_mask], G[label].y[test_mask])
                test_mse += mean_squared_error(test_true, test_pred)
                test_mae += mean_absolute_error(test_true, test_pred)
                test_r2 += r2_score(test_true, test_pred)

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

        epoch_losses.append(avg_test_loss)
        epoch_r2_scores.append(avg_test_r2)

        # Check for improvement
        if avg_test_loss < best_test_mae:
            best_test_mae = avg_test_loss
            best_test_r2 = avg_test_r2
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
        status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss**0.5:.4f}, Train R²: {avg_test_r2:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    return model, best_test_r2, best_test_mae**0.5, epoch_losses, epoch_r2_scores

# Streamlit configuration
st.set_page_config(
    page_title="Supply Chain Forecasting & GNN Classification",
    page_icon="📈",
    layout="wide"
)



class StreamlitTimeSeriesAnalyzer:
    def __init__(self, metadata_path: str = None, server_url: str = None, headers: Dict = None):
        """Initialize all forecasting models"""
        self.metadata_path ="./metadata.json"
        self.server_url = server_url
        self.headers = {"accept": "application/json"}

        # Initialize models only when needed
        self.models = {}
        
    def initialize_models(self):
        """Initialize all forecasting models"""
        if not self.models:
            self.models = {
                'single_step_arima': SingleStepARIMA(self.metadata_path),
                'multi_step_arima': MultiStepARIMA(self.metadata_path),
                'single_step_sarima': SingleStepSarimaAnalyzer(self.metadata_path),
                'multi_step_sarima': MultiStepSarimaAnalyzer(self.metadata_path),
                'single_step_cnnlstm': SingleStepCNNLSTMAnalyzer(self.metadata_path),
                'single_step_xgboost': XGBoostForecaster(self.metadata_path),
                'multi_step_xgboost': XGBoostForecaster(self.metadata_path),
                'single_step_prophet': SingleStepProphet(self.metadata_path),
                'multi_step_prophet': MultiStepProphet(self.metadata_path)
            }

# Plotting functions (from time series code)
def plot_single_step_forecast(train_demand, test_demand, demand_forecast, model_type, node_id, demand_mape, lookback=None):
    """Helper function to plot single-step forecasts"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(train_demand.index, train_demand.values, 
           label='Training Data', marker='o', linestyle='-', linewidth=2)
    ax.plot(test_demand.index, test_demand.values, 
           label='Test Data', marker='o', linestyle='-', linewidth=2)
    
    if model_type in ["CNN-LSTM", "XGBoost"]:
        lookback = lookback or 3
        forecast_index = test_demand.index[lookback:]
        if len(forecast_index) > len(demand_forecast):
            forecast_index = forecast_index[:len(demand_forecast)]
        elif len(forecast_index) < len(demand_forecast):
            demand_forecast = demand_forecast[:len(forecast_index)]
    else:
        forecast_index = test_demand.index
        
    ax.plot(forecast_index, demand_forecast, 
           label=f'{model_type} Forecast', marker='s', linestyle='--', linewidth=2)
    
    ax.set_title(f'Single-Step {model_type} Demand Forecast for Node {node_id}\nTest Set MAPE: {demand_mape:.2f}%')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_multi_step_forecast(train_demand, test_demand, forecasts, mapes, model_type, node_id, forecast_horizons, lookback=None):
    """Helper function to plot multi-step forecasts"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot training and test data
    ax.plot(train_demand.index, train_demand.values, 
           label='Training Data', marker='o', linestyle='-', linewidth=2, color='blue')
    ax.plot(test_demand.index, test_demand.values, 
           label='Test Data', marker='o', linestyle='-', linewidth=2, color='green')
    
    # Color map for different forecast horizons
    colors = plt.cm.rainbow(np.linspace(0, 1, len(forecast_horizons)))
    
    # Plot forecasts for each horizon
    for forecast, horizon, color in zip(forecasts, forecast_horizons, colors):
        # Adjust forecast index based on the model type and horizon
        if model_type == "XGBoost":
            lookback = lookback or 3
            start_index = train_demand.index[-1]
            forecast_index = pd.date_range(start=start_index, periods=len(forecast), freq=test_demand.index.freq)[1:]
        else:
            start_index = test_demand.index[horizon-1]
            forecast_index = pd.date_range(start=start_index, periods=len(forecast), freq=test_demand.index.freq)
        
        # Truncate forecast if needed
        if len(forecast_index) > len(forecast):
            forecast_index = forecast_index[:len(forecast)]
        elif len(forecast_index) < len(forecast):
            forecast = forecast[:len(forecast_index)]
        
        # Plot the forecast
        ax.plot(forecast_index, forecast, 
               label=f'{horizon}-Step Forecast (MAPE: {mapes[horizon]:.2f}%)',
               marker='s', linestyle='--', linewidth=2, color=color)
    
    ax.set_title(f'Multi-Step {model_type} Demand Forecast for Node {node_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Main function
def main():
    st.title("Supply Chain Forecasting & GNN Based Analysis")
    
    # Sidebar selection
    task = st.sidebar.selectbox(
        "Select Task",
        ["Time Series Forecasting", "GNN Based Analysis","Hybrid Model"]
    )
    
    if task == "Time Series Forecasting":
        st.subheader("Time Series Forecasting")
        
        # Data source selection
        data_source = st.sidebar.radio("Select Data Source", ["Local Directory", "Server"])
        version = st.sidebar.text_input("Enter Version of the fetch","NSS_1000_12",key="version")

        if data_source == "Local Directory":
            st.sidebar.header("Local Directory Settings")
            local_dir = st.sidebar.text_input("Enter local directory path", "./data/")
            try:

                parser = TemporalHeterogeneousGraphParser(
                    base_url="", 
                    version="", 
                    headers={"accept": "application/json"}, 
                    meta_data_path="./metadata.json", 
                    use_local_files=True, 
                    local_dir=local_dir+"/"+version+"/", 
                    num_classes = 20)
                st.sidebar.success("Successfully loaded local files!")
            except Exception as e:
                st.sidebar.error(f"Error loading local files: {str(e)}")
                return
                
        else:  # Server
            st.sidebar.header("Server Settings")
            server_url = os.getenv("SERVER_URL")
            # st.write(server_url)
            
            if server_url:
                version = st.sidebar.text_input("Enter Version of the fetch","NSS_1000_12",key="version")
                try:
                    parser = TemporalHeterogeneousGraphParser(
                        base_url=server_url, 
                        version=version, 
                        headers={"accept": "application/json"}, 
                        meta_data_path="./metadata.json", 
                        use_local_files=False, 
                        local_dir=local_dir+"/"+version+"/",  
                        num_classes = 20
                        )
                    st.sidebar.success("Successfully connected to server!")
                except Exception as e:
                    st.sidebar.error(f"Error connecting to server: {str(e)}")
                    return
            else:
                st.sidebar.warning("Please enter a server URL")
                return
        
        # Create temporal graph and get demand DataFrame
        try:
            with st.spinner("Loading and processing data..."):
                temporal_graphs, hetero_obj =parser.create_temporal_graph(regression = False, out_steps = 3, multistep = False, task = 'df', threshold=10)
                demand_df = parser.get_df()
                demand_df.index = pd.to_datetime(demand_df.index)
                
                # Initialize analyzer
                analyzer = StreamlitTimeSeriesAnalyzer(
                    metadata_path="./metadata.json" if data_source == "Local Directory" else None,
                    server_url=server_url if data_source == "Server" else None,
                    headers = {"accept": "application/json"} if data_source == "Server" else None
                )
                analyzer.initialize_models()
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return
        
        st.subheader("Demand Data Preview")
        st.dataframe(demand_df)
        
        # Model selection and parameters
        model_type = st.sidebar.radio("Select Model Type", ["ARIMA", "SARIMA", "CNN-LSTM", "XGBoost", "Prophet"])
        node_id = st.selectbox("Select Node for Analysis", demand_df.columns)
        analysis_type = st.radio("Select Forecasting Type", ["Single-Step", "Multi-Step"])
        
        # Model-specific parameters
        lookback = None
        if model_type == "CNN-LSTM":
            st.sidebar.subheader("CNN-LSTM Parameters")
            lookback = st.sidebar.slider("Lookback Period", 3, 10, 3)
            num_epochs = st.sidebar.slider("Number of Epochs", 50, 200, 100)
            batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)
        elif model_type == "XGBoost":
            st.sidebar.subheader("XGBoost Parameters")
            lookback = st.sidebar.slider("Lookback Period", 3, 10, 3)
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100)
            max_depth = st.sidebar.slider("Max Depth", 3, 10, 6)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
        
        if node_id:
            st.write(f"Analyzing Node: {node_id}")
            
            # Train-test split
            train_demand, test_demand = analyzer.models['single_step_arima'].train_test_split(demand_df[node_id])
            
            if analysis_type == "Single-Step":
                st.subheader(f"Single-Step {model_type} Forecasting")
                
                # Get forecasts based on model type
                if model_type == "ARIMA":
                    demand_forecast, demand_mape = analyzer.models['single_step_arima'].forecast_node(
                        train_demand, test_demand
                    )
                elif model_type == "SARIMA":
                    demand_forecast, demand_mape, order_params, seasonal_params = (
                        analyzer.models['single_step_sarima'].forecast_node_sarima(
                            train_demand, test_demand
                        )
                    )
                    st.write(f"SARIMA Parameters - Order: {order_params}, Seasonal: {seasonal_params}")
                elif model_type == "CNN-LSTM":
                    with st.spinner('Training CNN-LSTM model...'):
                        demand_forecast, demand_mape = analyzer.models['single_step_cnnlstm'].forecast_node_cnn_lstm(
                            train_demand, 
                            test_demand,
                            node_id,
                            'demand',
                            lookback=lookback,
                            epochs=num_epochs,
                            batch_size=batch_size
                        )
                elif model_type == "Prophet":
                    with st.spinner('Training Prophet model...'):
                        demand_forecast, demand_mape = analyzer.models['single_step_prophet'].forecast_node(
                            train_demand,
                            test_demand
                        )
                else:  # XGBoost
                    with st.spinner('Training XGBoost model...'):
                        demand_forecast, demand_mape = analyzer.models['single_step_xgboost'].single_step_forecast(
                            train_demand,
                            test_demand
                        )
                
                # Plot results
                fig = plot_single_step_forecast(
                    train_demand, test_demand, demand_forecast, 
                    model_type, node_id, demand_mape, lookback
                )
                st.pyplot(fig)
                plt.close()
                
                st.metric(f"{model_type} Demand Forecast MAPE", f"{demand_mape:.2f}%")
            
            elif analysis_type == "Multi-Step":
                st.subheader(f"Multi-Step {model_type} Forecasting")
                if model_type == "CNN-LSTM":
                    st.warning("Multi-step forecasting is not yet implemented for CNN-LSTM model")
                else:
                    forecast_horizons = st.multiselect(
                        "Select Forecast Horizons (Steps)", 
                        [1, 2, 3, 4, 5, 6], 
                        default=[1,3]
                    )
                    
                    if forecast_horizons:
                        if model_type == "ARIMA":
                            forecasts, mapes = analyzer.models['multi_step_arima'].forecast_node_multistep(
                                train_demand, test_demand, forecast_horizons
                            )
                        elif model_type == "SARIMA":
                            forecasts, mapes, order_params, seasonal_params = (
                                analyzer.models['multi_step_sarima'].forecast_node_multistep_sarima(
                                    train_demand, test_demand, forecast_horizons
                                )
                            )
                            st.write(f"SARIMA Parameters - Order: {order_params}, Seasonal: {seasonal_params}")
                        elif model_type == "Prophet":
                            # forecasts, mapes = analyzer.models['multi_step_prophet'].forecast_node_multistep(
                            #     train_demand, test_demand, forecast_horizons)
                            st.warning("No implemenation for multi-step prophet")
                        else:  # XGBoost
                            with st.spinner('Training XGBoost model...'):
                                forecasts, mapes = analyzer.models['multi_step_xgboost'].multi_step_forecast(
                                    train_demand,
                                    test_demand,
                                    forecast_horizons
                                )
                            
                            # Plot results
                            fig = plot_multi_step_forecast(
                                train_demand, test_demand, forecasts, mapes,
                                model_type, node_id, forecast_horizons, lookback
                            )
                            st.pyplot(fig)
                            plt.close()
                            
                            # Display MAPEs
                            col1, col2 = st.columns(2)
                            for i, horizon in enumerate(forecast_horizons):
                                if i % 2 == 0:
                                    col1.metric(f"{horizon}-Step Forecast MAPE", f"{mapes[horizon]:.2f}%")
                                else:
                                    col2.metric(f"{horizon}-Step Forecast MAPE", f"{mapes[horizon]:.2f}%")
        
        else:
            st.warning("Please select a node for analysis.")
    
    elif task == "GNN Based Analysis":
        st.subheader("Graph Neural Network Training Interface")
        # Model Configuration Section

        with st.sidebar.expander("🎯 Task Configuration", expanded=True):
            task_type = st.radio("Select Task Type", 
                                ["Classification", "Regression"], 
                                help="Choose whether to perform node classification or regression")
        
        # Store task type in session state
            st.session_state.task_type = task_type
            num_layers = st.number_input(
                "Number of Layers", 
                min_value=1, 
                max_value=5, 
                value=2,
                help="Number of graph convolutional layers"
            )
            layer_config = {}
            for i in range(num_layers):
                encoder_type = st.sidebar.selectbox(
                        "Select Layer Type",
                        ["SAGEConv", "GATConv" , "GeneralConv"] , key=f"layer_{i}"
                    )
                st.session_state[f'encoder_type_{i}'] = encoder_type

                # Layer-specific parameters based on encoder type
                if encoder_type == 'SAGEConv':
                    normalize = st.checkbox(f"Normalize Layer {i+1}", value=True)
                    layer_config[f'layer{i+1}'] = {
                        'normalize': normalize
                    }
                
                elif encoder_type == 'GATConv':
                    heads = st.number_input(f"Number of Attention Heads Layer {i+1}", 
                                            min_value=1, value=1)
                    dropout = st.slider(f"Dropout Layer {i+1}", 
                                        min_value=0.0, 
                                        max_value=1.0, 
                                        value=0.0)
                    layer_config[f'layer{i+1}'] = {
                        'heads': heads,
                        'dropout': dropout
                    }
                
                elif encoder_type == 'GeneralConv':
                    aggr = st.selectbox(f"Aggregation Type Layer {i+1}", 
                                        ['add', 'mean', 'max'])

                    attention = st.checkbox(f"Use Attention Layer {i+1}", value=False)
                    layer_config[f'layer{i+1}'] = {
                        'aggr': aggr,
                        'attention': attention
                    }
        with st.sidebar.expander("⚙️ Training Parameters", expanded=True):
            hidden_channels = st.number_input(
                "Hidden Channels", 
                min_value=8, 
                max_value=256, 
                value=64,
                help="Number of hidden channels in the graph neural network"
            )
            
            # num_parts = st.number_input(
            #     "Number of Parts", 
            #     min_value=10, 
            #     max_value=1000, 
            #     value=500,
            #     help="Number of parts for the GRU Decoder"
            # )
            
            num_epochs = st.number_input(
                "Number of Epochs", 
                min_value=1, 
                max_value=10000, 
                value=50,
                help="Number of training epochs"
            )
            
            learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.0001, 
                max_value=0.1, 
                value=0.001,
                format="%.4f",
                help="Learning rate for the optimizer"
            )
            
            patience = st.number_input(
                "Early Stopping Patience", 
                min_value=1, 
                max_value=50, 
                value=10,
                help="Number of epochs to wait before early stopping"
            )
        
        # Data Configuration
        with st.sidebar.expander("📊 Data Configuration", expanded=True):
            use_local_files = st.checkbox("Use Local Files", value=False)
            # metadata_path = ""
            
            if use_local_files:
                local_dir = st.text_input("Local Directory Path", "./data")
  
        
        # Start Training Button
        train_button = st.sidebar.button("🚀 Start Training")

        if train_button:
            try:
                # Initialize parser and create temporal graph
                base_url = os.getenv("SERVER_URL")
                version = st.sidebar.text_input("Enter Version of the fetch","NSS_1000_12",key="version")
                headers = {"accept": "application/json"}
                
                parser = TemporalHeterogeneousGraphParser(
                    base_url=base_url,
                    version=version,
                    headers = {"accept": "application/json"},
                    meta_data_path="metadata.json",
                    use_local_files=use_local_files,
                    local_dir=local_dir+"/", 
                )
                
                # Create temporal graphs based on task type
                regression_flag = True if st.session_state.task_type == "Regression" else False
                temporal_graphs, hetero_obj = parser.create_temporal_graph(regression = regression_flag, out_steps = 3, multistep = False, task = 'df', threshold=10)
                
                # Select the graph (e.g., at time step 10)
                G = temporal_graphs[10][1]
                
                # Initialize model
                model = Model3(
                    hidden_channels=hidden_channels, 
                    out_channels= 1 if task_type == "Regression" else len(temporal_graphs[1][1]["PARTS"].y), 
                    num_parts=len(temporal_graphs[1][1]["PARTS"].y),
                    G=G
                )
                
                # Move model to appropriate device
                device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                
                # Configure optimizer and loss function
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                if task_type == "Classification":
                    loss_fn = F.cross_entropy
                    train_fn = train_classification
                else:
                    loss_fn = F.mse_loss
                    train_fn = train_regression
                
                # Start training
                if task_type == "Classification":
                    trained_model, best_test_accuracy, best_test_loss, epoch_losses, epoch_accuracies = train_classification(
                        num_epochs, model, optimizer, loss_fn, temporal_graphs, label='PARTS', device=device, patience=patience
                    )
                else:
                    trained_model, best_test_r2, best_test_mae, epoch_losses, epoch_r2_scores = train_regression(
                        num_epochs, model, optimizer, loss_fn, temporal_graphs, label='PARTS', device=device, patience=patience
                    )
                
                # Display results
                if task_type == "Classification":
                    st.success(f"Training Completed. Best Test Accuracy: {best_test_accuracy:.4f}, Best Test Loss: {best_test_loss:.4f}")
                else:
                    st.success(f"Training Completed. Best R² Score: {best_test_r2:.4f}, Best MAE: {best_test_mae:.4f}")
                
                # Dashboard for Model Performance
                st.subheader("Model Performance Dashboard")
                
                # Custom CSS for styling
                st.markdown(
                    """
                    <style>
                    .metric-card {
                        
                        padding: 20px;
                        border-radius: 10px;
                        margin: 10px;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        transition: transform 0.3s ease-in-out;
                    }
                    .metric-card:hover {
                        transform: scale(1.05);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                
                # Create columns for metrics
           
                if task_type == "Classification":
                    col1, col2 = st.columns(2)
                    col1.markdown(f"<div class='metric-card'><h3>Accuracy</h3><p>{best_test_accuracy:.4f}</p></div>", unsafe_allow_html=True)
                    col2.markdown(f"<div class='metric-card'><h3>Loss</h3><p>{best_test_loss:.4f}</p></div>", unsafe_allow_html=True)
                    
                    # Plot performance graph for loss
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=list(range(len(epoch_losses))), y=epoch_losses, mode='lines+markers', name='Loss'))
                    fig_loss.update_layout(title="Epoch-wise Loss", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss)
                    
                    # Plot performance graph for accuracy
                    fig_accuracy = go.Figure()
                    fig_accuracy.add_trace(go.Scatter(x=list(range(len(epoch_accuracies))), y=epoch_accuracies, mode='lines+markers', name='Accuracy'))
                    fig_accuracy.update_layout(title="Epoch-wise Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                    st.plotly_chart(fig_accuracy)
                else:
                    col1, col2 = st.columns(2)
                    col1.markdown(f"<div class='metric-card'><h3>R² Score</h3><p>{best_test_r2:.4f}</p></div>", unsafe_allow_html=True)
                    col2.markdown(f"<div class='metric-card'><h3>MAE</h3><p>{best_test_mae:.4f}</p></div>", unsafe_allow_html=True)
                    
                    # Plot performance graph for loss
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=list(range(len(epoch_losses))), y=epoch_losses, mode='lines+markers', name='Loss'))
                    fig_loss.update_layout(title="Epoch-wise Loss", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_loss)
                    
                    # Plot performance graph for R² score
                    fig_r2 = go.Figure()
                    fig_r2.add_trace(go.Scatter(x=list(range(len(epoch_r2_scores))), y=epoch_r2_scores, mode='lines+markers', name='R² Score'))
                    fig_r2.update_layout(title="Epoch-wise R² Score", xaxis_title="Epoch", yaxis_title="R² Score")
                    st.plotly_chart(fig_r2)
            except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
                print(str(e))

    elif task == "Hybrid Model":
        st.subheader("Hybrid Model")
        # Data source selection
        data_source = st.sidebar.radio("Select Data Source", ["Local Directory", "Server"])
        version = st.sidebar.text_input("Enter Version of the fetch","NSS_1000_12",key="version")

        if data_source == "Local Directory":
            st.sidebar.header("Local Directory Settings")
            local_dir = st.sidebar.text_input("Enter local directory path", "./data/")
            try:

                parser = TemporalHeterogeneousGraphParser(
                    base_url="", 
                    version="", 
                    headers={"accept": "application/json"}, 
                    meta_data_path="./metadata.json", 
                    use_local_files=True, 
                    local_dir=local_dir+"/"+version+"/", 
                    num_classes = 20)
                st.sidebar.success("Successfully loaded local files!")
            except Exception as e:
                st.sidebar.error(f"Error loading local files: {str(e)}")
                return
                
        else:  # Server
            st.sidebar.header("Server Settings")
            server_url = os.getenv("SERVER_URL")
            # st.write(server_url)
            
            if server_url:
                version = st.sidebar.text_input("Enter Version of the fetch","NSS_1000_12",key="version")
                try:
                    parser = TemporalHeterogeneousGraphParser(
                        base_url=server_url, 
                        version=version, 
                        headers={"accept": "application/json"}, 
                        meta_data_path="./metadata.json", 
                        use_local_files=False, 
                        local_dir=local_dir+"/"+version+"/",  
                        num_classes = 20
                        )
                    st.sidebar.success("Successfully connected to server!")
                    
                except Exception as e:
                    st.sidebar.error(f"Error connecting to server: {str(e)}")
                    return
            else:
                st.sidebar.warning("Please enter a server URL")
                return
        try:
            with st.spinner("Loading and processing data..."):
                temporal_graphs, hetero_obj =parser.create_temporal_graph(regression = False, out_steps = 3, multistep = False, task = 'df', threshold=10)
                demand_df = parser.get_df()
                demand_df.index = pd.to_datetime(demand_df.index)
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return                
        
        model_choice = st.selectbox("Select a Type", ["Non-Aggregated Columns", "Aggregated Columns"])
        if model_choice == "Non-Aggregated Columns":
            part_id_list = []
            part_data = parser.get_extended_df()
            # st.dataframe(part_data)
            labels_df = parser.get_df()
            for x in labels_df.columns:
                part_data[x]['demand'] = labels_df[x]
            # st.dataframe(labels_df)
            
            for i in labels_df.columns:
                part_id_list.append(i)
            
            # final_agg = {}   
            node_id = st.selectbox("Select part id", labels_df.columns)
            if st.button("Run Forecasting"):
                viz,mape = hm.demand_forecasting(part_data,node_id)
                # final_agg = agg_mape
            # Display results
                st.write("Aggregated MAPE Scores:")
                # st.write(mape)
            
            # for key, values in final_agg.items():
            #     if node_id == key:
            #         plot = values[0]
            #         mape = values[1]
                    
                st.pyplot(viz)
                st.write(mape)
                
        elif model_choice == "Aggregated Columns":
            aggregation_method = st.radio("Select column aggregation type",["mean","sum","min","max"])
            part_id_list = []
            labels_df = parser.get_df()
            node_id = st.selectbox("Select part id", labels_df.columns)
            part_data = parser.aggregate_part_features(node_id,aggregation_method)
            
            part_data['demand'] = labels_df[node_id]
            
            st.dataframe(part_data)
            
            for i in labels_df.columns:
                part_id_list.append(i)
            
            # # final_agg = {}   
            # # node_id = st.selectbox("Select part id", labels_df.columns)
            if st.button("Run Forecasting"):
                viz,mape = hm.aggregated_demand_forecasting(part_data,node_id)
                # final_agg = agg_mape
            # Display results
                st.write("Aggregated MAPE Scores:")
                # st.write(mape)
            
            # # for key, values in final_agg.items():
            # #     if node_id == key:
            # #         plot = values[0]
            # #         mape = values[1]
                    
                st.pyplot(viz)
                st.write(mape)
# Run the main function
if __name__ == "__main__":
    main()