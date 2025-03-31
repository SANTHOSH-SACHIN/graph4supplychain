# # # # import streamlit as st
# # # # import os
# # # # import pandas as pd
# # # # import numpy as np
# # # # import torch
# # # # import torch.nn as nn
# # # # from torch.optim import Adam
# # # # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # # # from sklearn.preprocessing import StandardScaler
# # # # from torch_geometric.data import Data
# # # # from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv
# # # # import h5py
# # # # import pickle
# # # # import scipy.sparse as sp
# # # # import matplotlib.pyplot as plt
# # # # import time
# # # # import plotly.graph_objects as go
# # # # import plotly.express as px

# # # # st.set_page_config(page_title="Traffic Prediction GNN Model", layout="wide")

# # # # # --------------------------
# # # # # Set up the UI
# # # # # --------------------------
# # # # st.title("Traffic Prediction using Graph Neural Networks")

# # # # st.markdown("""
# # # # This application allows you to train and evaluate Graph Neural Network models
# # # # for traffic prediction on different datasets. You can adjust model parameters
# # # # and compare different GNN encoder types.
# # # # """)

# # # # # Sidebar for configuration options
# # # # st.sidebar.header("Model Configuration")

# # # # # --------------------------
# # # # # 1. Dataset Selection
# # # # # --------------------------
# # # # dataset_name = st.sidebar.selectbox(
# # # #     "Select Dataset",
# # # #     ["PeMSD7", "METR-LA", "Chickenpox"],
# # # #     index=0
# # # # )

# # # # st.sidebar.subheader("GNN Model Parameters")

# # # # # --------------------------
# # # # # 2. Encoder selection
# # # # # --------------------------
# # # # encoder_type = st.sidebar.selectbox(
# # # #     "Encoder Type",
# # # #     ["Transformer", "GAT", "SAGE", "General"],
# # # #     index=0
# # # # )

# # # # # --------------------------
# # # # # 3. Hyperparameters
# # # # # --------------------------
# # # # col1, col2 = st.sidebar.columns(2)
# # # # with col1:
# # # #     hidden_channels = st.number_input("Hidden Channels", min_value=16, max_value=256, value=64, step=16)
# # # #     batch_size = st.number_input("Batch Size", min_value=16, max_value=256, value=64, step=16)

# # # # with col2:
# # # #     epochs = st.number_input("Epochs", min_value=5, max_value=200, value=50, step=5)
# # # #     lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f", step=0.001)

# # # # # --------------------------
# # # # # 4. Dataset Handling
# # # # # --------------------------
# # # # @st.cache_data
# # # # def load_dataset(dataset_name):
# # # #     if dataset_name == 'PeMSD7':
# # # #         # Load PeMSD7 data
# # # #         try:
# # # #             speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
# # # #             adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
# # # #             edge_attr = None
# # # #             nodes = speed_data.shape[1]
# # # #             timesteps = speed_data.shape[0]
# # # #         except FileNotFoundError:
# # # #             st.error(f"PeMSD7 dataset files not found. Please ensure the files exist in the dataset folder.")
# # # #             return None, None, None

# # # #     elif dataset_name == 'METR-LA':
# # # #         # Load METR-LA data
# # # #         try:
# # # #             with h5py.File('dataset/metr-la.h5', 'r') as f:
# # # #                 speed_data = pd.DataFrame(f['df']['block0_values'][:])

# # # #             # Load adjacency matrix
# # # #             with open('dataset/adj_mx.pkl', 'rb') as f:
# # # #                 adj_mx = pickle.load(f, encoding='latin1')
# # # #             adjacency_matrix = adj_mx[2]
# # # #             nodes = speed_data.shape[1]
# # # #             timesteps = speed_data.shape[0]
# # # #         except FileNotFoundError:
# # # #             st.error(f"METR-LA dataset files not found. Please ensure the files exist in the dataset folder.")
# # # #             return None, None, None

# # # #     elif dataset_name == 'Chickenpox':
# # # #         # Load Chickenpox data
# # # #         try:
# # # #             speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
# # # #             adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
# # # #             edge_attr = None

# # # #             # Remove non-numeric columns (e.g., dates)
# # # #             speed_data = speed_data.select_dtypes(include=[np.number])
# # # #             nodes = speed_data.shape[1]
# # # #             timesteps = speed_data.shape[0]
# # # #         except FileNotFoundError:
# # # #             st.error(f"Chickenpox dataset files not found. Please ensure the files exist in the dataset folder.")
# # # #             return None, None, None

# # # #     # Common preprocessing
# # # #     scaler = StandardScaler()
# # # #     speed_data_scaled = scaler.fit_transform(speed_data)

# # # #     # Create edge_index and edge_attr
# # # #     if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
# # # #         edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
# # # #         edge_attr = None
# # # #     else:
# # # #         adj_sparse = sp.coo_matrix(adjacency_matrix)
# # # #         edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
# # # #         edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

# # # #     # Store the information in a dictionary
# # # #     dataset_info = {
# # # #         'speed_data': speed_data,
# # # #         'speed_data_scaled': speed_data_scaled,
# # # #         'adjacency_matrix': adjacency_matrix,
# # # #         'scaler': scaler,
# # # #         'nodes': nodes,
# # # #         'timesteps': timesteps
# # # #     }

# # # #     return speed_data_scaled, edge_index, edge_attr, dataset_info

# # # # # --------------------------
# # # # # 5. Model Definition
# # # # # --------------------------
# # # # class MLP(nn.Module):
# # # #     def __init__(self, in_channels, hidden_layers, out_channels):
# # # #         super().__init__()
# # # #         layers = []
# # # #         prev = in_channels
# # # #         for h in hidden_layers:
# # # #             layers.extend([nn.Linear(prev, h), nn.ReLU()])
# # # #             prev = h
# # # #         layers.append(nn.Linear(prev, out_channels))
# # # #         self.mlp = nn.Sequential(*layers)

# # # #     def forward(self, x):
# # # #         return self.mlp(x)

# # # # class GRUDecoder(nn.Module):
# # # #     def __init__(self, in_channels, out_channels, hidden_gru=64):
# # # #         super().__init__()
# # # #         self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
# # # #         self.mlp = MLP(hidden_gru, [128, 64, 32], out_channels)

# # # #     def forward(self, z):
# # # #         gru_out, _ = self.gru(z)
# # # #         return self.mlp(gru_out.squeeze(1))

# # # # class GNNModel(nn.Module):
# # # #     def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None, encoder_type='Transformer'):
# # # #         super().__init__()
# # # #         self.edge_index = edge_index
# # # #         self.edge_attr = edge_attr

# # # #         # Encoder selection
# # # #         if encoder_type == 'SAGE':
# # # #             self.encoder = nn.Sequential(
# # # #                 SAGEConv((-1, -1), hidden_channels),
# # # #                 nn.ReLU(),
# # # #                 SAGEConv((-1, -1), hidden_channels)
# # # #             )
# # # #         elif encoder_type == 'GAT':
# # # #             self.encoder = nn.Sequential(
# # # #                 GATConv((-1, -1), hidden_channels, add_self_loops=False),
# # # #                 nn.ReLU(),
# # # #                 GATConv((-1, -1), hidden_channels, add_self_loops=False)
# # # #             )
# # # #         elif encoder_type == 'General':
# # # #             self.encoder = nn.Sequential(
# # # #                 GeneralConv((-1, -1), hidden_channels),
# # # #                 nn.ReLU(),
# # # #                 GeneralConv((-1, -1), hidden_channels)
# # # #             )
# # # #         elif encoder_type == 'Transformer':
# # # #             edge_dim = edge_attr.size(1) if edge_attr is not None else None
# # # #             self.encoder = nn.Sequential(
# # # #                 TransformerConv(input_size, hidden_channels, edge_dim=edge_dim),
# # # #                 nn.ReLU(),
# # # #                 TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
# # # #             )
# # # #         else:
# # # #             raise ValueError(f"Unknown encoder type: {encoder_type}")

# # # #         self.decoder = GRUDecoder(hidden_channels, out_channels)

# # # #     def forward(self, x):
# # # #         if isinstance(self.encoder[0], TransformerConv):
# # # #             encoded = self.encoder[0](x, self.edge_index, self.edge_attr)
# # # #             encoded = self.encoder[1](encoded)
# # # #             encoded = self.encoder[2](encoded, self.edge_index, self.edge_attr)
# # # #         else:
# # # #             encoded = self.encoder[0](x, self.edge_index)
# # # #             encoded = self.encoder[1](encoded)
# # # #             encoded = self.encoder[2](encoded, self.edge_index)
# # # #         return self.decoder(encoded.unsqueeze(1))

# # # # # --------------------------
# # # # # 6. Training & Evaluation
# # # # # --------------------------
# # # # def train_evaluate(speed_data, edge_index, edge_attr, dataset_info, encoder_type, hidden_channels, batch_size, epochs, lr):
# # # #     # Training progress containers
# # # #     progress_bar = st.progress(0)
# # # #     loss_container = st.empty()
# # # #     metrics_container = st.empty()

# # # #     # Prepare training metrics tracking
# # # #     train_losses = []

# # # #     input_size = speed_data.shape[1]

# # # #     # Convert to tensors
# # # #     tensor_data = torch.tensor(speed_data, dtype=torch.float32)
# # # #     train_size = int(len(tensor_data) * 0.8)
# # # #     train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]

# # # #     # Create model
# # # #     model = GNNModel(
# # # #         input_size=input_size,
# # # #         hidden_channels=hidden_channels,
# # # #         out_channels=input_size,
# # # #         edge_index=edge_index,
# # # #         edge_attr=edge_attr,
# # # #         encoder_type=encoder_type
# # # #     )
# # # #     optimizer = Adam(model.parameters(), lr=lr)
# # # #     loss_fn = nn.MSELoss()

# # # #     # Initialize metrics display
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         train_loss_chart = st.empty()
# # # #     with col2:
# # # #         eval_metrics_chart = st.empty()

# # # #     # Training
# # # #     start_time = time.time()

# # # #     for epoch in range(epochs):
# # # #         model.train()
# # # #         optimizer.zero_grad()
# # # #         output = model(train_data[:-1])
# # # #         loss = loss_fn(output, train_data[1:])
# # # #         loss.backward()
# # # #         optimizer.step()

# # # #         # Update progress
# # # #         progress_bar.progress((epoch + 1) / epochs)
# # # #         train_losses.append(loss.item())

# # # #         # Display realtime training info
# # # #         elapsed = time.time() - start_time
# # # #         loss_container.markdown(f"**Epoch {epoch+1}/{epochs}** • Loss: {loss.item():.4f} • Time: {elapsed:.2f}s")

# # # #         # Update loss chart every few epochs
# # # #         if epoch % 2 == 0 or epoch == epochs - 1:
# # # #             fig = px.line(x=list(range(1, len(train_losses)+1)), y=train_losses,
# # # #                         labels={'x': 'Epoch', 'y': 'Loss'}, title='Training Loss')
# # # #             train_loss_chart.plotly_chart(fig)

# # # #     # Evaluation
# # # #     model.eval()
# # # #     with torch.no_grad():
# # # #         output = model(test_data[:-1])
# # # #         mse = mean_squared_error(test_data[1:].numpy(), output.numpy())
# # # #         mae = mean_absolute_error(test_data[1:].numpy(), output.numpy())
# # # #         rmse = np.sqrt(mse)
# # # #         mape = np.mean(np.abs((test_data[1:].numpy() - output.numpy()) / (test_data[1:].numpy() + 1e-10)))

# # # #         # Display final metrics
# # # #         metrics_data = pd.DataFrame({
# # # #             'Metric': ['MSE', 'MAE', 'RMSE', 'MAPE'],
# # # #             'Value': [mse, mae, rmse, mape]
# # # #         })

# # # #         eval_metrics_chart.plotly_chart(
# # # #             px.bar(metrics_data, x='Metric', y='Value', title='Evaluation Metrics')
# # # #         )

# # # #         metrics_container.markdown(f"""
# # # #         ## Evaluation Results
# # # #         - **MSE**: {mse:.4f}
# # # #         - **MAE**: {mae:.4f}
# # # #         - **RMSE**: {rmse:.4f}
# # # #         - **MAPE**: {mape:.4f}
# # # #         """)

# # # #     # Visualize predictions
# # # #     st.subheader("Prediction vs. Ground Truth")
# # # #     sample_node = st.slider("Select node to visualize", 0, input_size-1, 0)

# # # #     # Select a limited window of data points for visualization
# # # #     window_size = 100
# # # #     pred_data = output.numpy()[-window_size:, sample_node]
# # # #     actual_data = test_data[1:].numpy()[-window_size:, sample_node]

# # # #     # Scale back to original values
# # # #     scaler = dataset_info['scaler']

# # # #     # Create visualization dataframe
# # # #     vis_df = pd.DataFrame({
# # # #         'Timestep': list(range(len(pred_data))),
# # # #         'Predicted': pred_data,
# # # #         'Actual': actual_data
# # # #     })

# # # #     # Plot using Plotly
# # # #     fig = go.Figure()
# # # #     fig.add_trace(go.Scatter(x=vis_df['Timestep'], y=vis_df['Actual'],
# # # #                              mode='lines', name='Ground Truth', line=dict(color='blue')))
# # # #     fig.add_trace(go.Scatter(x=vis_df['Timestep'], y=vis_df['Predicted'],
# # # #                              mode='lines', name='Predicted', line=dict(color='red')))
# # # #     fig.update_layout(title=f'Traffic Prediction for Node {sample_node}',
# # # #                      xaxis_title='Timestep',
# # # #                      yaxis_title='Scaled Value')
# # # #     st.plotly_chart(fig, use_container_width=True)

# # # #     return model

# # # # # --------------------------
# # # # # Main application flow
# # # # # --------------------------
# # # # # Show dataset information
# # # # if dataset_name:
# # # #     with st.spinner(f"Loading {dataset_name} dataset..."):
# # # #         data_load = load_dataset(dataset_name)

# # # #         if data_load is not None and len(data_load) == 4:
# # # #             speed_data, edge_index, edge_attr, dataset_info = data_load

# # # #             # Display dataset information
# # # #             st.header("Dataset Information")
# # # #             col1, col2, col3 = st.columns(3)
# # # #             with col1:
# # # #                 st.metric("Number of Nodes", dataset_info['nodes'])
# # # #             with col2:
# # # #                 st.metric("Number of Timesteps", dataset_info['timesteps'])
# # # #             with col3:
# # # #                 if edge_attr is not None:
# # # #                     st.metric("Number of Edges", edge_index.shape[1])
# # # #                 else:
# # # #                     st.metric("Number of Edges", edge_index.shape[1])

# # # #             # Display sample of the data
# # # #             with st.expander("View data sample"):
# # # #                 st.dataframe(dataset_info['speed_data'].head())

# # # #             # Display graph visualization
# # # #             with st.expander("Network Graph Visualization"):
# # # #                 st.warning("Full graph visualization not shown due to computational complexity. Displaying sample information.")
# # # #                 st.text(f"Total Number of Edges: {edge_index.shape[1]}")
# # # #                 st.text(f"Edge Index Shape: {edge_index.shape}")
# # # #                 if edge_attr is not None:
# # # #                     st.text(f"Edge Attribute Shape: {edge_attr.shape}")
# # # #         else:
# # # #             st.error("Failed to load dataset. Please check if the dataset files exist.")

# # # # # Start training when button is pressed
# # # # if st.sidebar.button("Train Model"):
# # # #     if data_load is not None and len(data_load) == 4:
# # # #         st.header(f"Training {encoder_type} Model on {dataset_name}")

# # # #         model = train_evaluate(
# # # #             speed_data,
# # # #             edge_index,
# # # #             edge_attr,
# # # #             dataset_info,
# # # #             encoder_type=encoder_type,
# # # #             hidden_channels=hidden_channels,
# # # #             batch_size=batch_size,
# # # #             epochs=epochs,
# # # #             lr=lr
# # # #         )
# # # #     else:
# # # #         st.error("Please make sure the dataset is loaded correctly before training.")

# # # # # Add information about the app
# # # # with st.sidebar.expander("About this app"):
# # # #     st.markdown("""
# # # #     This application allows you to experiment with different Graph Neural Network architectures
# # # #     for traffic prediction tasks. It supports multiple encoder types and datasets.

# # # #     **Supported Encoders:**
# # # #     - Transformer - TransformerConv layers
# # # #     - GAT - Graph Attention Network
# # # #     - SAGE - GraphSAGE implementation
# # # #     - General - General Graph Convolution

# # # #     **Datasets:**
# # # #     - PeMSD7 - California highway data
# # # #     - METR-LA - Los Angeles traffic data
# # # #     - Chickenpox - Hungary chickenpox cases
# # # #     """)

# # # import streamlit as st
# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # from torch.optim import Adam
# # # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # # from sklearn.preprocessing import StandardScaler
# # # from torch_geometric.data import Data
# # # from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv
# # # import h5py
# # # import pickle
# # # import scipy.sparse as sp
# # # import matplotlib.pyplot as plt
# # # import time
# # # from datetime import datetime
# # # import plotly.graph_objects as go
# # # import plotly.express as px

# # # st.set_page_config(
# # #     page_title="AGNet - AdaptiveGraphNet",
# # #     page_icon="🌐",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # --------------------------
# # # # Streamlit UI Components
# # # # --------------------------
# # # st.title("AGNet: Adaptive Graph Neural Network for Spatio-Temporal Data")
# # # st.markdown("""
# # # This application allows you to train and evaluate AGNet models on various spatio-temporal datasets.
# # # AGNet uses an Encoder-Temporal Decoder architecture to capture both spatial and temporal dependencies.
# # # """)

# # # # Create sidebar for model configuration
# # # st.sidebar.header("Model Configuration")

# # # # Dataset selection
# # # dataset_name = st.sidebar.selectbox(
# # #     "Dataset",
# # #     ["PeMSD7", "METR-LA", "Chickenpox"],
# # #     help="Select the dataset for training and evaluation"
# # # )

# # # # Encoder Type
# # # encoder_type = st.sidebar.selectbox(
# # #     "Encoder Type",
# # #     ["Transformer", "GAT", "SAGE", "General"],
# # #     help="Select the graph neural network encoder architecture"
# # # )

# # # # Training parameters
# # # st.sidebar.header("Training Parameters")
# # # epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10,
# # #                          help="Number of training epochs")
# # # lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f",
# # #                            help="Learning rate for optimizer")
# # # batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=64, step=16,
# # #                              help="Batch size for training (Note: Currently using full sequence training)")
# # # train_split = st.sidebar.slider("Training Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
# # #                               help="Proportion of data used for training vs testing")

# # # # Model architecture parameters
# # # st.sidebar.header("Model Architecture")
# # # hidden_channels = st.sidebar.slider("Hidden Channels", min_value=16, max_value=256, value=64, step=16,
# # #                                   help="Number of hidden channels in the GNN encoder")
# # # hidden_gru = st.sidebar.slider("GRU Hidden Size", min_value=16, max_value=256, value=64, step=16,
# # #                              help="Hidden size for GRU decoder")
# # # mlp_layers = st.sidebar.multiselect("MLP Decoder Layers", options=[16, 32, 64, 128, 256], default=[128, 64, 32],
# # #                                   help="Hidden layers for MLP decoder")

# # # # Optimizer options
# # # optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"],
# # #                                     help="Optimizer for training")
# # # weight_decay = st.sidebar.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0, step=0.0001, format="%.5f",
# # #                                      help="L2 regularization strength")

# # # # --------------------------
# # # # 1. Dataset Handling
# # # # --------------------------
# # # @st.cache_data
# # # def load_dataset(dataset_name):
# # #     with st.spinner(f"Loading {dataset_name} dataset..."):
# # #         if dataset_name == 'PeMSD7':
# # #             # Load PeMSD7 data
# # #             try:
# # #                 speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
# # #                 adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
# # #                 edge_attr = None
# # #                 nodes = speed_data.shape[1]
# # #                 timesteps = speed_data.shape[0]
# # #                 st.info(f"PeMSD7 dataset loaded: {nodes} nodes, {timesteps} timesteps")
# # #             except FileNotFoundError:
# # #                 st.error("PeMSD7 dataset files not found. Please ensure PeMSD7_V_228.csv and PeMSD7_W_228.csv are in the dataset folder.")
# # #                 return None, None, None, None

# # #         elif dataset_name == 'METR-LA':
# # #             # Load METR-LA data
# # #             try:
# # #                 with h5py.File('dataset/metr-la.h5', 'r') as f:
# # #                     speed_data = pd.DataFrame(f['df']['block0_values'][:])

# # #                 # Load adjacency matrix
# # #                 with open('dataset/adj_mx.pkl', 'rb') as f:
# # #                     adj_mx = pickle.load(f, encoding='latin1')
# # #                 adjacency_matrix = adj_mx[2]
# # #                 nodes = speed_data.shape[1]
# # #                 timesteps = speed_data.shape[0]
# # #                 st.info(f"METR-LA dataset loaded: {nodes} nodes, {timesteps} timesteps")
# # #             except FileNotFoundError:
# # #                 st.error("METR-LA dataset files not found. Please ensure metr-la.h5 and adj_mx.pkl are in the dataset folder.")
# # #                 return None, None, None, None

# # #         elif dataset_name == 'Chickenpox':
# # #             # Load Chickenpox data
# # #             try:
# # #                 speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
# # #                 adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
# # #                 edge_attr = None

# # #                 # Remove non-numeric columns (e.g., dates)
# # #                 speed_data = speed_data.select_dtypes(include=[np.number])
# # #                 nodes = speed_data.shape[1]
# # #                 timesteps = speed_data.shape[0]
# # #                 st.info(f"Chickenpox dataset loaded: {nodes} nodes, {timesteps} timesteps")
# # #             except FileNotFoundError:
# # #                 st.error("Chickenpox dataset files not found. Please ensure hungary_chickenpox.csv and hungary_county_edges.csv are in the dataset folder.")
# # #                 return None, None, None, None

# # #         # Common preprocessing
# # #         scaler = StandardScaler()
# # #         normalized_data = scaler.fit_transform(speed_data)

# # #         # Create edge_index and edge_attr
# # #         if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
# # #             edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
# # #             edge_attr = None
# # #         else:
# # #             adj_sparse = sp.coo_matrix(adjacency_matrix)
# # #             edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
# # #             edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

# # #         return normalized_data, edge_index, edge_attr, scaler

# # # # --------------------------
# # # # 2. Model Definition
# # # # --------------------------
# # # class MLP(nn.Module):
# # #     def __init__(self, in_channels, hidden_layers, out_channels):
# # #         super().__init__()
# # #         layers = []
# # #         prev = in_channels
# # #         for h in hidden_layers:
# # #             layers.extend([nn.Linear(prev, h), nn.ReLU()])
# # #             prev = h
# # #         layers.append(nn.Linear(prev, out_channels))
# # #         self.mlp = nn.Sequential(*layers)

# # #     def forward(self, x):
# # #         return self.mlp(x)

# # # class GRUDecoder(nn.Module):
# # #     def __init__(self, in_channels, out_channels, hidden_gru=64, mlp_layers=[128, 64, 32]):
# # #         super().__init__()
# # #         self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
# # #         self.mlp = MLP(hidden_gru, mlp_layers, out_channels)

# # #     def forward(self, z):
# # #         gru_out, _ = self.gru(z)
# # #         return self.mlp(gru_out.squeeze(1))

# # # class GNNModel(nn.Module):
# # #     def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None,
# # #                  encoder_type='Transformer', hidden_gru=64, mlp_layers=[128, 64, 32]):
# # #         super().__init__()
# # #         self.edge_index = edge_index
# # #         self.edge_attr = edge_attr

# # #         # Encoder selection
# # #         if encoder_type == 'SAGE':
# # #             self.encoder = nn.Sequential(
# # #                 SAGEConv((-1, -1), hidden_channels),
# # #                 nn.ReLU(),
# # #                 SAGEConv((-1, -1), hidden_channels)
# # #             )
# # #         elif encoder_type == 'GAT':
# # #             self.encoder = nn.Sequential(
# # #                 GATConv((-1, -1), hidden_channels, add_self_loops=False),
# # #                 nn.ReLU(),
# # #                 GATConv((-1, -1), hidden_channels, add_self_loops=False)
# # #             )
# # #         elif encoder_type == 'General':
# # #             self.encoder = nn.Sequential(
# # #                 GeneralConv((-1, -1), hidden_channels),
# # #                 nn.ReLU(),
# # #                 GeneralConv((-1, -1), hidden_channels)
# # #             )
# # #         elif encoder_type == 'Transformer':
# # #             edge_dim = edge_attr.size(1) if edge_attr is not None else None
# # #             self.encoder = nn.Sequential(
# # #                 TransformerConv(input_size, hidden_channels, edge_dim=edge_dim),
# # #                 nn.ReLU(),
# # #                 TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
# # #             )
# # #         else:
# # #             raise ValueError(f"Unknown encoder type: {encoder_type}")

# # #         self.decoder = GRUDecoder(hidden_channels, out_channels, hidden_gru, mlp_layers)

# # #     def forward(self, x):
# # #         if isinstance(self.encoder[0], TransformerConv):
# # #             encoded = self.encoder[0](x, self.edge_index, self.edge_attr)
# # #             encoded = self.encoder[1](encoded)
# # #             encoded = self.encoder[2](encoded, self.edge_index, self.edge_attr)
# # #         else:
# # #             encoded = self.encoder[0](x, self.edge_index)
# # #             encoded = self.encoder[1](encoded)
# # #             encoded = self.encoder[2](encoded, self.edge_index)
# # #         return self.decoder(encoded.unsqueeze(1))

# # # # --------------------------
# # # # 3. Training & Evaluation
# # # # --------------------------
# # # def train_evaluate(normalized_data, edge_index, edge_attr, scaler,
# # #                    encoder_type='Transformer', epochs=50, lr=0.001,
# # #                    hidden_channels=64, train_split=0.8, hidden_gru=64,
# # #                    mlp_layers=[128, 64, 32], optimizer_name='Adam', weight_decay=0.0):
# # #     # Convert to tensors
# # #     tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
# # #     input_size = tensor_data.shape[1]

# # #     train_size = int(len(tensor_data) * train_split)
# # #     train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]

# # #     # Create model
# # #     model = GNNModel(
# # #         input_size=input_size,
# # #         hidden_channels=hidden_channels,
# # #         out_channels=input_size,
# # #         edge_index=edge_index,
# # #         edge_attr=edge_attr,
# # #         encoder_type=encoder_type,
# # #         hidden_gru=hidden_gru,
# # #         mlp_layers=mlp_layers
# # #     )

# # #     # Select optimizer
# # #     if optimizer_name == 'Adam':
# # #         optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# # #     elif optimizer_name == 'SGD':
# # #         optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
# # #     elif optimizer_name == 'RMSprop':
# # #         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

# # #     loss_fn = nn.MSELoss()

# # #     # Training stats
# # #     train_losses = []
# # #     progress_bar = st.progress(0)
# # #     status_text = st.empty()
# # #     loss_chart = st.empty()
# # #     start_time = time.time()

# # #     # Create a placeholder for the metrics
# # #     metrics_container = st.container()

# # #     # Training loop
# # #     for epoch in range(epochs):
# # #         model.train()
# # #         optimizer.zero_grad()
# # #         output = model(train_data[:-1])
# # #         loss = loss_fn(output, train_data[1:])
# # #         loss.backward()
# # #         optimizer.step()

# # #         train_losses.append(loss.item())

# # #         # Update progress
# # #         progress_bar.progress((epoch + 1) / epochs)
# # #         elapsed = time.time() - start_time
# # #         estimated_total = elapsed / (epoch + 1) * epochs
# # #         remaining = estimated_total - elapsed

# # #         status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
# # #                          f"Time elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")

# # #         # Update loss chart every few epochs
# # #         if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
# # #             fig = px.line(
# # #                 x=list(range(1, len(train_losses) + 1)),
# # #                 y=train_losses,
# # #                 labels={'x': 'Epoch', 'y': 'Loss'},
# # #                 title='Training Loss'
# # #             )
# # #             loss_chart.plotly_chart(fig, use_container_width=True)

# # #     # Evaluation
# # #     with torch.no_grad():
# # #         model.eval()
# # #         test_output = model(test_data[:-1])

# # #         # Inverse transform for true metrics
# # #         test_pred = test_output.numpy()
# # #         test_true = test_data[1:].numpy()

# # #         # Convert back to original scale if needed
# # #         if scaler:
# # #             # This is an approximation as we'd need the original data to invert properly
# # #             test_pred_original = scaler.inverse_transform(test_pred)
# # #             test_true_original = scaler.inverse_transform(test_true)
# # #         else:
# # #             test_pred_original = test_pred
# # #             test_true_original = test_true

# # #         # Calculate metrics
# # #         mse = mean_squared_error(test_true, test_pred)
# # #         mae = mean_absolute_error(test_true, test_pred)
# # #         rmse = np.sqrt(mse)

# # #         # MAPE with handling for zeros (avoid division by zero)
# # #         epsilon = 1e-10  # Small constant to avoid division by zero
# # #         mape = np.mean(np.abs((test_true - test_pred) / (np.abs(test_true) + epsilon))) * 100

# # #         # Display metrics
# # #         with metrics_container:
# # #             st.header("Model Performance")
# # #             col1, col2, col3, col4 = st.columns(4)
# # #             col1.metric("MSE", f"{mse:.4f}")
# # #             col2.metric("MAE", f"{mae:.4f}")
# # #             col3.metric("RMSE", f"{rmse:.4f}")
# # #             col4.metric("MAPE (%)", f"{mape:.2f}")

# # #             # Visualize predictions vs actual for a sample node
# # #             st.subheader("Predictions vs Actual")
# # #             sample_node = st.slider("Select node to visualize", 0, test_true.shape[1] - 1, 0)

# # #             # Create prediction visualization
# # #             fig = go.Figure()
# # #             fig.add_trace(go.Scatter(
# # #                 y=test_true_original[:, sample_node],
# # #                 mode='lines',
# # #                 name='Actual'
# # #             ))
# # #             fig.add_trace(go.Scatter(
# # #                 y=test_pred_original[:, sample_node],
# # #                 mode='lines',
# # #                 name='Prediction'
# # #             ))
# # #             fig.update_layout(
# # #                 title=f"Predictions vs Actual for Node {sample_node}",
# # #                 xaxis_title="Time Step",
# # #                 yaxis_title="Value",
# # #                 legend_title="Legend",
# # #                 hovermode="x unified"
# # #             )
# # #             st.plotly_chart(fig, use_container_width=True)

# # #             # Create error heatmap
# # #             error = np.abs(test_true - test_pred)

# # #             # Take a subset of the error matrix if it's too large
# # #             max_display = 50
# # #             if error.shape[0] > max_display or error.shape[1] > max_display:
# # #                 time_slice = slice(0, min(error.shape[0], max_display))
# # #                 node_slice = slice(0, min(error.shape[1], max_display))
# # #                 error_display = error[time_slice, node_slice]
# # #                 st.info(f"Displaying error heatmap for first {error_display.shape[0]} timesteps and {error_display.shape[1]} nodes.")
# # #             else:
# # #                 error_display = error

# # #             fig = px.imshow(
# # #                 error_display,
# # #                 labels=dict(x="Node", y="Time Step", color="Absolute Error"),
# # #                 title="Prediction Error Heatmap"
# # #             )
# # #             st.plotly_chart(fig, use_container_width=True)

# # #     return model, {
# # #         'mse': mse,
# # #         'mae': mae,
# # #         'rmse': rmse,
# # #         'mape': mape,
# # #         'train_losses': train_losses
# # #     }

# # # # --------------------------
# # # # 4. Main App Execution
# # # # --------------------------
# # # if st.button("Train Model"):
# # #     with st.spinner("Loading dataset..."):
# # #         normalized_data, edge_index, edge_attr, scaler = load_dataset(dataset_name)

# # #     if normalized_data is not None:
# # #         st.subheader(f"Training AGNet with {encoder_type} encoder on {dataset_name} dataset")

# # #         # Convert mlp_layers to integers
# # #         mlp_layers_int = [int(layer) for layer in mlp_layers]

# # #         try:
# # #             model, metrics = train_evaluate(
# # #                 normalized_data,
# # #                 edge_index,
# # #                 edge_attr,
# # #                 scaler,
# # #                 encoder_type=encoder_type,
# # #                 epochs=epochs,
# # #                 lr=lr,
# # #                 hidden_channels=hidden_channels,
# # #                 train_split=train_split,
# # #                 hidden_gru=hidden_gru,
# # #                 mlp_layers=mlp_layers_int,
# # #                 optimizer_name=optimizer_name,
# # #                 weight_decay=weight_decay
# # #             )

# # #             # Create download button for model
# # #             if st.button("Save Model"):
# # #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # #                 model_path = f"agnet_{dataset_name}_{encoder_type}_{timestamp}.pt"
# # #                 torch.save(model.state_dict(), model_path)
# # #                 st.success(f"Model saved to {model_path}")

# # #         except Exception as e:
# # #             st.error(f"Error during model training: {str(e)}")
# # #             st.exception(e)
# # # else:
# # #     st.info("Click 'Train Model' to start training AGNet on the selected dataset and parameters.")

# # # # Show explanations
# # # with st.expander("About AGNet Architecture"):
# # #     st.markdown("""
# # #     ## AGNet: Adaptive Graph Neural Network

# # #     AGNet is a spatio-temporal graph neural network designed for capturing both spatial dependencies through graph convolutions and temporal dependencies through a GRU-based decoder.

# # #     ### Architecture Components:

# # #     1. **Graph Neural Network Encoder**
# # #        - Captures spatial relationships between nodes in the graph
# # #        - Options: Transformer, GAT (Graph Attention), GraphSAGE, General Convolution

# # #     2. **Temporal GRU Decoder**
# # #        - Processes the encoded graph representations to capture temporal dependencies
# # #        - Uses GRU (Gated Recurrent Unit) followed by MLP layers

# # #     3. **Prediction Head**
# # #        - Multi-layer perceptron that maps GRU outputs to predictions

# # #     ### Applications:

# # #     - Traffic forecasting (PeMSD7, METR-LA datasets)
# # #     - Disease propagation modeling (Chickenpox dataset)
# # #     - Can be adapted for other spatio-temporal forecasting tasks
# # #     """)

# # # with st.expander("Dataset Information"):
# # #     st.markdown("""
# # #     ### PeMSD7
# # #     Traffic speed measurements collected from loop detectors in the California Performance Measurement System (PeMS) District 7, covering Los Angeles area freeways.

# # #     ### METR-LA
# # #     Traffic speed data collected from loop detectors in Los Angeles County highway system. Contains measurements from 207 sensors over 4 months.

# # #     ### Hungary Chickenpox
# # #     Weekly chickenpox cases reported in 20 counties of Hungary from 2005 to 2015. The graph is constructed based on geographical adjacency between counties.
# # #     """)



# # import streamlit as st
# # import os
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # from torch.optim import Adam
# # from sklearn.metrics import mean_absolute_error, mean_squared_error
# # from sklearn.preprocessing import StandardScaler
# # from torch_geometric.data import Data
# # from torch_geometric.nn import SAGEConv, GATConv, GeneralConv, TransformerConv
# # import h5py
# # import pickle
# # import scipy.sparse as sp
# # import matplotlib.pyplot as plt
# # import time
# # from datetime import datetime
# # import plotly.graph_objects as go
# # import plotly.express as px

# # st.set_page_config(
# #     page_title="AGNet - AdaptiveGraphNet",
# #     page_icon="🌐",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # --------------------------
# # # Streamlit UI Components
# # # --------------------------
# # st.title("AGNet: Adaptive Graph Neural Network for Spatio-Temporal Data")
# # st.markdown("""
# # This application allows you to train and evaluate AGNet models on various spatio-temporal datasets.
# # AGNet uses an Encoder-Temporal Decoder architecture to capture both spatial and temporal dependencies.
# # """)

# # # Create sidebar for model configuration
# # st.sidebar.header("Model Configuration")

# # # Dataset selection
# # dataset_name = st.sidebar.selectbox(
# #     "Dataset",
# #     ["PeMSD7", "METR-LA", "Chickenpox"],
# #     help="Select the dataset for training and evaluation"
# # )

# # # Encoder Type
# # encoder_type = st.sidebar.selectbox(
# #     "Encoder Type",
# #     ["Transformer", "GAT", "SAGE", "General"],
# #     help="Select the graph neural network encoder architecture"
# # )

# # # Training parameters
# # st.sidebar.header("Training Parameters")
# # epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10,
# #                          help="Number of training epochs")
# # lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f",
# #                            help="Learning rate for optimizer")
# # batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=64, step=16,
# #                              help="Batch size for training (Note: Currently using full sequence training)")
# # train_split = st.sidebar.slider("Training Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
# #                               help="Proportion of data used for training vs testing")

# # # Model architecture parameters
# # st.sidebar.header("Model Architecture")
# # hidden_channels = st.sidebar.slider("Hidden Channels", min_value=16, max_value=256, value=64, step=16,
# #                                   help="Number of hidden channels in the GNN encoder")
# # hidden_gru = st.sidebar.slider("GRU Hidden Size", min_value=16, max_value=256, value=64, step=16,
# #                              help="Hidden size for GRU decoder")
# # mlp_layers = st.sidebar.multiselect("MLP Decoder Layers", options=[16, 32, 64, 128, 256], default=[128, 64, 32],
# #                                   help="Hidden layers for MLP decoder")

# # # Optimizer options
# # optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"],
# #                                     help="Optimizer for training")
# # weight_decay = st.sidebar.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0, step=0.0001, format="%.5f",
# #                                      help="L2 regularization strength")

# # # Add node selection to sidebar
# # st.sidebar.header("Visualization Options")
# # # We'll initialize with 0, but will update the max value after loading the dataset
# # sample_node = st.sidebar.number_input("Node to visualize", min_value=0, value=0, step=1,
# #                                     help="Select which node to visualize in the predictions chart")

# # # --------------------------
# # # 1. Dataset Handling
# # # --------------------------
# # @st.cache_data
# # def load_dataset(dataset_name):
# #     with st.spinner(f"Loading {dataset_name} dataset..."):
# #         if dataset_name == 'PeMSD7':
# #             # Load PeMSD7 data
# #             try:
# #                 speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
# #                 adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
# #                 edge_attr = None
# #                 nodes = speed_data.shape[1]
# #                 timesteps = speed_data.shape[0]
# #                 st.info(f"PeMSD7 dataset loaded: {nodes} nodes, {timesteps} timesteps")
# #             except FileNotFoundError:
# #                 st.error("PeMSD7 dataset files not found. Please ensure PeMSD7_V_228.csv and PeMSD7_W_228.csv are in the dataset folder.")
# #                 return None, None, None, None, None

# #         elif dataset_name == 'METR-LA':
# #             # Load METR-LA data
# #             try:
# #                 with h5py.File('dataset/metr-la.h5', 'r') as f:
# #                     speed_data = pd.DataFrame(f['df']['block0_values'][:])

# #                 # Load adjacency matrix
# #                 with open('dataset/adj_mx.pkl', 'rb') as f:
# #                     adj_mx = pickle.load(f, encoding='latin1')
# #                 adjacency_matrix = adj_mx[2]
# #                 nodes = speed_data.shape[1]
# #                 timesteps = speed_data.shape[0]
# #                 st.info(f"METR-LA dataset loaded: {nodes} nodes, {timesteps} timesteps")
# #             except FileNotFoundError:
# #                 st.error("METR-LA dataset files not found. Please ensure metr-la.h5 and adj_mx.pkl are in the dataset folder.")
# #                 return None, None, None, None, None

# #         elif dataset_name == 'Chickenpox':
# #             # Load Chickenpox data
# #             try:
# #                 speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
# #                 adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
# #                 edge_attr = None

# #                 # Remove non-numeric columns (e.g., dates)
# #                 speed_data = speed_data.select_dtypes(include=[np.number])
# #                 nodes = speed_data.shape[1]
# #                 timesteps = speed_data.shape[0]
# #                 st.info(f"Chickenpox dataset loaded: {nodes} nodes, {timesteps} timesteps")
# #             except FileNotFoundError:
# #                 st.error("Chickenpox dataset files not found. Please ensure hungary_chickenpox.csv and hungary_county_edges.csv are in the dataset folder.")
# #                 return None, None, None, None, None

# #         # Common preprocessing
# #         scaler = StandardScaler()
# #         normalized_data = scaler.fit_transform(speed_data)

# #         # Create edge_index and edge_attr
# #         if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
# #             edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
# #             edge_attr = None
# #         else:
# #             adj_sparse = sp.coo_matrix(adjacency_matrix)
# #             edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
# #             edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

# #         return normalized_data, edge_index, edge_attr, scaler, nodes

# # # --------------------------
# # # 2. Model Definition
# # # --------------------------
# # class MLP(nn.Module):
# #     def __init__(self, in_channels, hidden_layers, out_channels):
# #         super().__init__()
# #         layers = []
# #         prev = in_channels
# #         for h in hidden_layers:
# #             layers.extend([nn.Linear(prev, h), nn.ReLU()])
# #             prev = h
# #         layers.append(nn.Linear(prev, out_channels))
# #         self.mlp = nn.Sequential(*layers)

# #     def forward(self, x):
# #         return self.mlp(x)

# # class GRUDecoder(nn.Module):
# #     def __init__(self, in_channels, out_channels, hidden_gru=64, mlp_layers=[128, 64, 32]):
# #         super().__init__()
# #         self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
# #         self.mlp = MLP(hidden_gru, mlp_layers, out_channels)

# #     def forward(self, z):
# #         gru_out, _ = self.gru(z)
# #         return self.mlp(gru_out.squeeze(1))

# # class GNNModel(nn.Module):
# #     def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None,
# #                  encoder_type='Transformer', hidden_gru=64, mlp_layers=[128, 64, 32]):
# #         super().__init__()
# #         self.edge_index = edge_index
# #         self.edge_attr = edge_attr

# #         # Encoder selection
# #         if encoder_type == 'SAGE':
# #             self.encoder = nn.Sequential(
# #                 SAGEConv((-1, -1), hidden_channels),
# #                 nn.ReLU(),
# #                 SAGEConv((-1, -1), hidden_channels)
# #             )
# #         elif encoder_type == 'GAT':
# #             self.encoder = nn.Sequential(
# #                 GATConv((-1, -1), hidden_channels, add_self_loops=False),
# #                 nn.ReLU(),
# #                 GATConv((-1, -1), hidden_channels, add_self_loops=False)
# #             )
# #         elif encoder_type == 'General':
# #             self.encoder = nn.Sequential(
# #                 GeneralConv((-1, -1), hidden_channels),
# #                 nn.ReLU(),
# #                 GeneralConv((-1, -1), hidden_channels)
# #             )
# #         elif encoder_type == 'Transformer':
# #             edge_dim = edge_attr.size(1) if edge_attr is not None else None
# #             self.encoder = nn.Sequential(
# #                 TransformerConv(input_size, hidden_channels, edge_dim=edge_dim),
# #                 nn.ReLU(),
# #                 TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
# #             )
# #         else:
# #             raise ValueError(f"Unknown encoder type: {encoder_type}")

# #         self.decoder = GRUDecoder(hidden_channels, out_channels, hidden_gru, mlp_layers)

# #     def forward(self, x):
# #         if isinstance(self.encoder[0], TransformerConv):
# #             encoded = self.encoder[0](x, self.edge_index, self.edge_attr)
# #             encoded = self.encoder[1](encoded)
# #             encoded = self.encoder[2](encoded, self.edge_index, self.edge_attr)
# #         else:
# #             encoded = self.encoder[0](x, self.edge_index)
# #             encoded = self.encoder[1](encoded)
# #             encoded = self.encoder[2](encoded, self.edge_index)
# #         return self.decoder(encoded.unsqueeze(1))

# # # --------------------------
# # # 3. Training & Evaluation
# # # --------------------------
# # def train_evaluate(normalized_data, edge_index, edge_attr, scaler, sample_node=0,
# #                    encoder_type='Transformer', epochs=50, lr=0.001,
# #                    hidden_channels=64, train_split=0.8, hidden_gru=64,
# #                    mlp_layers=[128, 64, 32], optimizer_name='Adam', weight_decay=0.0):
# #     # Convert to tensors
# #     tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
# #     input_size = tensor_data.shape[1]

# #     train_size = int(len(tensor_data) * train_split)
# #     train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]

# #     # Create model
# #     model = GNNModel(
# #         input_size=input_size,
# #         hidden_channels=hidden_channels,
# #         out_channels=input_size,
# #         edge_index=edge_index,
# #         edge_attr=edge_attr,
# #         encoder_type=encoder_type,
# #         hidden_gru=hidden_gru,
# #         mlp_layers=mlp_layers
# #     )

# #     # Select optimizer
# #     if optimizer_name == 'Adam':
# #         optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# #     elif optimizer_name == 'SGD':
# #         optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
# #     elif optimizer_name == 'RMSprop':
# #         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

# #     loss_fn = nn.MSELoss()

# #     # Training stats
# #     train_losses = []
# #     progress_bar = st.progress(0)
# #     status_text = st.empty()
# #     loss_chart = st.empty()
# #     start_time = time.time()

# #     # Create a placeholder for the metrics
# #     metrics_container = st.container()

# #     # Training loop
# #     for epoch in range(epochs):
# #         model.train()
# #         optimizer.zero_grad()
# #         output = model(train_data[:-1])
# #         loss = loss_fn(output, train_data[1:])
# #         loss.backward()
# #         optimizer.step()

# #         train_losses.append(loss.item())

# #         # Update progress
# #         progress_bar.progress((epoch + 1) / epochs)
# #         elapsed = time.time() - start_time
# #         estimated_total = elapsed / (epoch + 1) * epochs
# #         remaining = estimated_total - elapsed

# #         status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
# #                          f"Time elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")

# #         # Update loss chart every few epochs
# #         if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
# #             fig = px.line(
# #                 x=list(range(1, len(train_losses) + 1)),
# #                 y=train_losses,
# #                 labels={'x': 'Epoch', 'y': 'Loss'},
# #                 title='Training Loss'
# #             )
# #             loss_chart.plotly_chart(fig, use_container_width=True)

# #     # Evaluation
# #     with torch.no_grad():
# #         model.eval()
# #         test_output = model(test_data[:-1])

# #         # Inverse transform for true metrics
# #         test_pred = test_output.numpy()
# #         test_true = test_data[1:].numpy()

# #         # Convert back to original scale if needed
# #         if scaler:
# #             # This is an approximation as we'd need the original data to invert properly
# #             test_pred_original = scaler.inverse_transform(test_pred)
# #             test_true_original = scaler.inverse_transform(test_true)
# #         else:
# #             test_pred_original = test_pred
# #             test_true_original = test_true

# #         # Calculate metrics
# #         mse = mean_squared_error(test_true, test_pred)
# #         mae = mean_absolute_error(test_true, test_pred)
# #         rmse = np.sqrt(mse)

# #         # MAPE with handling for zeros (avoid division by zero)
# #         epsilon = 1e-10  # Small constant to avoid division by zero
# #         mape = np.mean(np.abs((test_true - test_pred) / (np.abs(test_true) + epsilon))) * 100

# #         # Display metrics
# #         with metrics_container:
# #             st.header("Model Performance")
# #             col1, col2, col3, col4 = st.columns(4)
# #             col1.metric("MSE", f"{mse:.4f}")
# #             col2.metric("MAE", f"{mae:.4f}")
# #             col3.metric("RMSE", f"{rmse:.4f}")
# #             col4.metric("MAPE (%)", f"{mape:.2f}")

# #             # Use the sample_node that was selected in the sidebar
# #             st.subheader(f"Predictions vs Actual for Node {sample_node}")

# #             # Create prediction visualization
# #             fig = go.Figure()
# #             fig.add_trace(go.Scatter(
# #                 y=test_true_original[:, sample_node],
# #                 mode='lines',
# #                 name='Actual'
# #             ))
# #             fig.add_trace(go.Scatter(
# #                 y=test_pred_original[:, sample_node],
# #                 mode='lines',
# #                 name='Prediction'
# #             ))
# #             fig.update_layout(
# #                 title=f"Predictions vs Actual for Node {sample_node}",
# #                 xaxis_title="Time Step",
# #                 yaxis_title="Value",
# #                 legend_title="Legend",
# #                 hovermode="x unified"
# #             )
# #             st.plotly_chart(fig, use_container_width=True)

# #             # Create error heatmap
# #             error = np.abs(test_true - test_pred)

# #             # Take a subset of the error matrix if it's too large
# #             max_display = 50
# #             if error.shape[0] > max_display or error.shape[1] > max_display:
# #                 time_slice = slice(0, min(error.shape[0], max_display))
# #                 node_slice = slice(0, min(error.shape[1], max_display))
# #                 error_display = error[time_slice, node_slice]
# #                 st.info(f"Displaying error heatmap for first {error_display.shape[0]} timesteps and {error_display.shape[1]} nodes.")
# #             else:
# #                 error_display = error

# #             fig = px.imshow(
# #                 error_display,
# #                 labels=dict(x="Node", y="Time Step", color="Absolute Error"),
# #                 title="Prediction Error Heatmap"
# #             )
# #             st.plotly_chart(fig, use_container_width=True)

# #     return model, {
# #         'mse': mse,
# #         'mae': mae,
# #         'rmse': rmse,
# #         'mape': mape,
# #         'train_losses': train_losses
# #     }

# # # --------------------------
# # # 4. Main App Execution
# # # --------------------------
# # # First, let's load the dataset to get the number of nodes
# # with st.spinner("Loading dataset..."):
# #     normalized_data, edge_index, edge_attr, scaler, num_nodes = load_dataset(dataset_name)

# # # Update the max value for the node selection
# # if num_nodes is not None:
# #     # Update the max value of the node selection widget
# #     st.sidebar.write(f"Total nodes: {num_nodes}")
# #     # Ensure the selected node is within the valid range
# #     if sample_node >= num_nodes:
# #         sample_node = num_nodes - 1
# #         st.sidebar.warning(f"Selected node was out of range. Adjusted to {sample_node}.")

# # if st.button("Train Model"):
# #     if normalized_data is not None and num_nodes is not None:
# #         st.subheader(f"Training AGNet with {encoder_type} encoder on {dataset_name} dataset")

# #         # Convert mlp_layers to integers
# #         mlp_layers_int = [int(layer) for layer in mlp_layers]

# #         try:
# #             model, metrics = train_evaluate(
# #                 normalized_data,
# #                 edge_index,
# #                 edge_attr,
# #                 scaler,
# #                 sample_node=sample_node,  # Pass the selected node
# #                 encoder_type=encoder_type,
# #                 epochs=epochs,
# #                 lr=lr,
# #                 hidden_channels=hidden_channels,
# #                 train_split=train_split,
# #                 hidden_gru=hidden_gru,
# #                 mlp_layers=mlp_layers_int,
# #                 optimizer_name=optimizer_name,
# #                 weight_decay=weight_decay
# #             )

# #             # Create download button for model
# #             if st.button("Save Model"):
# #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# #                 model_path = f"agnet_{dataset_name}_{encoder_type}_{timestamp}.pt"
# #                 torch.save(model.state_dict(), model_path)
# #                 st.success(f"Model saved to {model_path}")

# #         except Exception as e:
# #             st.error(f"Error during model training: {str(e)}")
# #             st.exception(e)
# # else:
# #     st.info("Click 'Train Model' to start training AGNet on the selected dataset and parameters.")

# # # Show explanations
# # with st.expander("About AGNet Architecture"):
# #     st.markdown("""
# #     ## AGNet: Adaptive Graph Neural Network

# #     AGNet is a spatio-temporal graph neural network designed for capturing both spatial dependencies through graph convolutions and temporal dependencies through a GRU-based decoder.

# #     ### Architecture Components:

# #     1. **Graph Neural Network Encoder**
# #        - Captures spatial relationships between nodes in the graph
# #        - Options: Transformer, GAT (Graph Attention), GraphSAGE, General Convolution

# #     2. **Temporal GRU Decoder**
# #        - Processes the encoded graph representations to capture temporal dependencies
# #        - Uses GRU (Gated Recurrent Unit) followed by MLP layers

# #     3. **Prediction Head**
# #        - Multi-layer perceptron that maps GRU outputs to predictions

# #     ### Applications:

# #     - Traffic forecasting (PeMSD7, METR-LA datasets)
# #     - Disease propagation modeling (Chickenpox dataset)
# #     - Can be adapted for other spatio-temporal forecasting tasks
# #     """)

# # with st.expander("Dataset Information"):
# #     st.markdown("""
# #     ### PeMSD7
# #     Traffic speed measurements collected from loop detectors in the California Performance Measurement System (PeMS) District 7, covering Los Angeles area freeways.

# #     ### METR-LA
# #     Traffic speed data collected from loop detectors in Los Angeles County highway system. Contains measurements from 207 sensors over 4 months.

# #     ### Hungary Chickenpox
# #     Weekly chickenpox cases reported in 20 counties of Hungary from 2005 to 2015. The graph is constructed based on geographical adjacency between counties.
# #     """)

# import streamlit as st
# import os
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
# import matplotlib.pyplot as plt
# import time
# from datetime import datetime
# import plotly.graph_objects as go
# import plotly.express as px

# st.set_page_config(
#     page_title="AGNet - AdaptiveGraphNet",
#     page_icon="🌐",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --------------------------
# # Streamlit UI Components
# # --------------------------
# st.title("AGNet: Adaptive Graph Neural Network for Spatio-Temporal Data 🌐")
# st.markdown("""
# This application allows you to train and evaluate AGNet models on various spatio-temporal datasets.
# AGNet uses an Encoder-Temporal Decoder architecture to capture both spatial and temporal dependencies.
# """)

# # Create sidebar for model configuration
# st.sidebar.header("Model Configuration")

# # Dataset selection
# dataset_name = st.sidebar.selectbox(
#     "Dataset",
#     ["PeMSD7", "METR-LA", "Chickenpox"],
#     help="Select the dataset for training and evaluation"
# )

# # Encoder Type
# encoder_type = st.sidebar.selectbox(
#     "Encoder Type",
#     ["Transformer", "GAT", "SAGE", "General"],
#     help="Select the graph neural network encoder architecture"
# )

# # Training parameters
# st.sidebar.header("Training Parameters")
# epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10,
#                          help="Number of training epochs")
# lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f",
#                            help="Learning rate for optimizer")
# batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=64, step=16,
#                              help="Batch size for training (Note: Currently using full sequence training)")
# train_split = st.sidebar.slider("Training Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
#                               help="Proportion of data used for training vs testing")

# # Note about batch size
# st.sidebar.info("Note: The model is currently trained on the full sequence without batching. The batch size parameter is not used.")

# # Model architecture parameters
# st.sidebar.header("Model Architecture")
# hidden_channels = st.sidebar.slider("Hidden Channels", min_value=16, max_value=256, value=64, step=16,
#                                   help="Number of hidden channels in the GNN encoder")
# hidden_gru = st.sidebar.slider("GRU Hidden Size", min_value=16, max_value=256, value=64, step=16,
#                              help="Hidden size for GRU decoder")
# mlp_layers = st.sidebar.multiselect("MLP Decoder Layers", options=[16, 32, 64, 128, 256], default=[128, 64, 32],
#                                   help="Hidden layers for MLP decoder")

# # Optimizer options
# optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"],
#                                     help="Optimizer for training")
# weight_decay = st.sidebar.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0, step=0.0001, format="%.5f",
#                                      help="L2 regularization strength")

# # Add node selection to sidebar
# st.sidebar.header("Visualization Options")
# # We'll initialize with 0, but will update the max value after loading the dataset
# sample_node = st.sidebar.number_input("Node to visualize", min_value=0, value=0, step=1,
#                                     help="Select which node to visualize in the predictions chart")

# # --------------------------
# # 1. Dataset Handling
# # --------------------------
# @st.cache_data
# def load_dataset(dataset_name):
#     with st.spinner(f"Loading {dataset_name} dataset..."):
#         if dataset_name == 'PeMSD7':
#             # Load PeMSD7 data
#             try:
#                 speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
#                 adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
#                 edge_attr = None
#                 nodes = speed_data.shape[1]
#                 timesteps = speed_data.shape[0]
#                 st.info(f"PeMSD7 dataset loaded: {nodes} nodes, {timesteps} timesteps")
#             except FileNotFoundError:
#                 st.error("PeMSD7 dataset files not found. Please ensure PeMSD7_V_228.csv and PeMSD7_W_228.csv are in the dataset folder.")
#                 return None, None, None, None, None

#         elif dataset_name == 'METR-LA':
#             # Load METR-LA data
#             try:
#                 with h5py.File('dataset/metr-la.h5', 'r') as f:
#                     speed_data = pd.DataFrame(f['df']['block0_values'][:])

#                 # Load adjacency matrix
#                 with open('dataset/adj_mx.pkl', 'rb') as f:
#                     adj_mx = pickle.load(f, encoding='latin1')
#                 adjacency_matrix = adj_mx[2]
#                 nodes = speed_data.shape[1]
#                 timesteps = speed_data.shape[0]
#                 st.info(f"METR-LA dataset loaded: {nodes} nodes, {timesteps} timesteps")
#             except FileNotFoundError:
#                 st.error("METR-LA dataset files not found. Please ensure metr-la.h5 and adj_mx.pkl are in the dataset folder.")
#                 return None, None, None, None, None

#         elif dataset_name == 'Chickenpox':
#             # Load Chickenpox data
#             try:
#                 speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
#                 adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
#                 edge_attr = None

#                 # Remove non-numeric columns (e.g., dates)
#                 speed_data = speed_data.select_dtypes(include=[np.number])
#                 nodes = speed_data.shape[1]
#                 timesteps = speed_data.shape[0]
#                 st.info(f"Chickenpox dataset loaded: {nodes} nodes, {timesteps} timesteps")
#             except FileNotFoundError:
#                 st.error("Chickenpox dataset files not found. Please ensure hungary_chickenpox.csv and hungary_county_edges.csv are in the dataset folder.")
#                 return None, None, None, None, None

#         # Common preprocessing
#         scaler = StandardScaler()
#         normalized_data = scaler.fit_transform(speed_data)

#         # Create edge_index and edge_attr
#         if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
#             edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
#             edge_attr = None
#         else:
#             adj_sparse = sp.coo_matrix(adjacency_matrix)
#             edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
#             edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

#         return normalized_data, edge_index, edge_attr, scaler, nodes

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
#     def __init__(self, in_channels, out_channels, hidden_gru=64, mlp_layers=[128, 64, 32]):
#         super().__init__()
#         self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
#         self.mlp = MLP(hidden_gru, mlp_layers, out_channels)

#     def forward(self, z):
#         gru_out, _ = self.gru(z)
#         return self.mlp(gru_out.squeeze(1))

# class GNNModel(nn.Module):
#     def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None,
#                  encoder_type='Transformer', hidden_gru=64, mlp_layers=[128, 64, 32]):
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

#         self.decoder = GRUDecoder(hidden_channels, out_channels, hidden_gru, mlp_layers)

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
# def train_evaluate(normalized_data, edge_index, edge_attr, scaler, sample_node=0,
#                    encoder_type='Transformer', epochs=50, lr=0.001,
#                    hidden_channels=64, train_split=0.8, hidden_gru=64,
#                    mlp_layers=[128, 64, 32], optimizer_name='Adam', weight_decay=0.0):
#     # Convert to tensors
#     tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
#     input_size = tensor_data.shape[1]

#     train_size = int(len(tensor_data) * train_split)
#     train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]

#     # Create model
#     model = GNNModel(
#         input_size=input_size,
#         hidden_channels=hidden_channels,
#         out_channels=input_size,
#         edge_index=edge_index,
#         edge_attr=edge_attr,
#         encoder_type=encoder_type,
#         hidden_gru=hidden_gru,
#         mlp_layers=mlp_layers
#     )

#     # Select optimizer
#     if optimizer_name == 'Adam':
#         optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif optimizer_name == 'SGD':
#         optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
#     elif optimizer_name == 'RMSprop':
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

#     loss_fn = nn.MSELoss()

#     # Training stats
#     train_losses = []
#     progress_bar = st.progress(0)
#     status_text = st.empty()
#     loss_chart = st.empty()
#     start_time = time.time()

#     # Create a placeholder for the metrics
#     metrics_container = st.container()

#     # Training loop
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         output = model(train_data[:-1])
#         loss = loss_fn(output, train_data[1:])
#         loss.backward()
#         optimizer.step()

#         train_losses.append(loss.item())

#         # Update progress
#         progress_bar.progress((epoch + 1) / epochs)
#         elapsed = time.time() - start_time
#         estimated_total = elapsed / (epoch + 1) * epochs
#         remaining = estimated_total - elapsed

#         status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
#                          f"Time elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")

#         # Update loss chart every few epochs
#         if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
#             fig = px.line(
#                 x=list(range(1, len(train_losses) + 1)),
#                 y=train_losses,
#                 labels={'x': 'Epoch', 'y': 'Loss'},
#                 title='Training Loss',
#                 template='plotly_white'
#             )
#             fig.update_traces(line_color='#0033CC')
#             loss_chart.plotly_chart(fig, use_container_width=True)

#     # Evaluation
#     with torch.no_grad():
#         model.eval()
#         test_output = model(test_data[:-1])

#         # Inverse transform for true metrics
#         test_pred = test_output.numpy()
#         test_true = test_data[1:].numpy()

#         # Convert back to original scale if needed
#         if scaler:
#             test_pred_original = scaler.inverse_transform(test_pred)
#             test_true_original = scaler.inverse_transform(test_true)
#         else:
#             test_pred_original = test_pred
#             test_true_original = test_true

#         # Calculate metrics
#         mse = mean_squared_error(test_true, test_pred)
#         mae = mean_absolute_error(test_true, test_pred)
#         rmse = np.sqrt(mse)

#         # MAPE with handling for zeros (avoid division by zero)
#         epsilon = 1e-10  # Small constant to avoid division by zero
#         mape = np.mean(np.abs((test_true - test_pred) / (np.abs(test_true) + epsilon))) * 100

#         # Display metrics
#         with metrics_container:
#             st.header("Model Performance")
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("MSE", f"{mse:.4f}")
#             col2.metric("MAE", f"{mae:.4f}")
#             col3.metric("RMSE", f"{rmse:.4f}")
#             col4.metric("MAPE (%)", f"{mape:.2f}")

#             # Use the sample_node that was selected in the sidebar
#             st.subheader(f"Predictions vs Actual for Node {sample_node}")

#             # Create prediction visualization
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(
#                 y=test_true_original[:, sample_node],
#                 mode='lines',
#                 name='Actual',
#                 line=dict(color='#0033CC')
#             ))
#             fig.add_trace(go.Scatter(
#                 y=test_pred_original[:, sample_node],
#                 mode='lines',
#                 name='Prediction',
#                 line=dict(color='#FF3333')
#             ))
#             fig.update_layout(
#                 title=f"Predictions vs Actual for Node {sample_node}",
#                 xaxis_title="Time Step",
#                 yaxis_title="Value",
#                 legend_title="Legend",
#                 hovermode="x unified",
#                 template='plotly_white'
#             )
#             st.plotly_chart(fig, use_container_width=True)

#             # Create error heatmap
#             error = np.abs(test_true - test_pred)

#             # Take a subset of the error matrix if it's too large
#             max_display = 50
#             if error.shape[0] > max_display or error.shape[1] > max_display:
#                 time_slice = slice(0, min(error.shape[0], max_display))
#                 node_slice = slice(0, min(error.shape[1], max_display))
#                 error_display = error[time_slice, node_slice]
#                 st.info(f"Displaying error heatmap for first {error_display.shape[0]} timesteps and {error_display.shape[1]} nodes.")
#             else:
#                 error_display = error

#             fig = px.imshow(
#                 error_display,
#                 labels=dict(x="Node", y="Time Step", color="Absolute Error"),
#                 title="Prediction Error Heatmap",
#                 color_continuous_scale='Blues'
#             )
#             st.plotly_chart(fig, use_container_width=True)

#     return model, {
#         'mse': mse,
#         'mae': mae,
#         'rmse': rmse,
#         'mape': mape,
#         'train_losses': train_losses
#     }

# # --------------------------
# # 4. Main App Execution
# # --------------------------
# # First, let's load the dataset to get the number of nodes
# with st.spinner("Loading dataset..."):
#     normalized_data, edge_index, edge_attr, scaler, num_nodes = load_dataset(dataset_name)

# # Update the max value for the node selection
# if num_nodes is not None:
#     # Update the max value of the node selection widget
#     st.sidebar.write(f"Total nodes: {num_nodes}")
#     # Ensure the selected node is within the valid range
#     if sample_node >= num_nodes:
#         sample_node = num_nodes - 1
#         st.sidebar.warning(f"Selected node was out of range. Adjusted to {sample_node}.")

# if st.button("Train Model"):
#     if normalized_data is not None and num_nodes is not None:
#         st.subheader(f"Training AGNet with {encoder_type} encoder on {dataset_name} dataset")

#         # Convert mlp_layers to integers
#         mlp_layers_int = [int(layer) for layer in mlp_layers]

#         try:
#             model, metrics = train_evaluate(
#                 normalized_data,
#                 edge_index,
#                 edge_attr,
#                 scaler,
#                 sample_node=sample_node,  # Pass the selected node
#                 encoder_type=encoder_type,
#                 epochs=epochs,
#                 lr=lr,
#                 hidden_channels=hidden_channels,
#                 train_split=train_split,
#                 hidden_gru=hidden_gru,
#                 mlp_layers=mlp_layers_int,
#                 optimizer_name=optimizer_name,
#                 weight_decay=weight_decay
#             )

#             # Create download button for model
#             if st.button("Save Model"):
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 model_path = f"agnet_{dataset_name}_{encoder_type}_{timestamp}.pt"
#                 torch.save(model.state_dict(), model_path)
#                 st.success(f"Model saved to {model_path}")

#         except Exception as e:
#             st.error(f"Error during model training: {str(e)}")
#             st.exception(e)
# else:
#     st.info("Click 'Train Model' to start training AGNet on the selected dataset and parameters.")

# # Show explanations
# with st.expander("About AGNet Architecture"):
#     st.markdown("""
#     ## AGNet: Adaptive Graph Neural Network

#     AGNet is a spatio-temporal graph neural network designed for capturing both spatial dependencies through graph convolutions and temporal dependencies through a GRU-based decoder.

#     ### Architecture Components:

#     1. **Graph Neural Network Encoder**
#        - Captures spatial relationships between nodes in the graph
#        - Options: Transformer, GAT (Graph Attention), GraphSAGE, General Convolution

#     2. **Temporal GRU Decoder**
#        - Processes the encoded graph representations to capture temporal dependencies
#        - Uses GRU (Gated Recurrent Unit) followed by MLP layers

#     3. **Prediction Head**
#        - Multi-layer perceptron that maps GRU outputs to predictions

#     ### Applications:

#     - Traffic forecasting (PeMSD7, METR-LA datasets)
#     - Disease propagation modeling (Chickenpox dataset)
#     - Can be adapted for other spatio-temporal forecasting tasks
#     """)

# with st.expander("Dataset Information"):
#     st.markdown("""
#     ### PeMSD7 🚗
#     Traffic speed measurements collected from loop detectors in the California Performance Measurement System (PeMS) District 7, covering Los Angeles area freeways.

#     ### METR-LA 🚦
#     Traffic speed data collected from loop detectors in Los Angeles County highway system. Contains measurements from 207 sensors over 4 months.

#     ### Hungary Chickenpox 🐔
#     Weekly chickenpox cases reported in 20 counties of Hungary from 2005 to 2015. The graph is constructed based on geographical adjacency between counties.
#     """)

import streamlit as st
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
import matplotlib.pyplot as plt
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# --------------------------
# Custom CSS for improved styling and light mode
# --------------------------
st.set_page_config(
    page_title="AGNet - AdaptiveGraphNet",
    page_icon="\U0001f310",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for light mode and better contrasting colours
custom_css = """
<style>
/* Force light background for main container */
.reportview-container, .main {
    background-color: #f9f9f9;
    color: #333;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-image: linear-gradient(#f9f9f9, #eaeaea);
    color: #333;
}

/* Buttons */
.stButton>button {
    background-color: #007BFF;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5em 1em;
}
.stButton>button:hover {
    background-color: #0056b3;
}

/* Progress bar customization */
.stProgress > div > div {
    background-color: #007BFF;
}

/* Override Plotly figures default background */
.plotly-graph-div {
    background: #f9f9f9 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --------------------------
# Utility: Load Lottie Animation
# --------------------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error("Error loading animation")
        return None

# Load a sample lottie animation
lottie_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_t9gkkhz4.json")

# --------------------------
# Streamlit UI Components
# --------------------------
st.title("AGNet: Adaptive Graph Neural Network for Spatio-Temporal Data")
st.markdown("""
This application allows you to train and evaluate AGNet models on various spatio-temporal datasets.
AGNet uses an Encoder-Temporal Decoder architecture to capture both spatial and temporal dependencies.
""")

# Display the lottie animation if loaded successfully
if lottie_animation:
    st_lottie(lottie_animation, height=200, key="lottie")

# Create sidebar for model configuration
st.sidebar.header("Model Configuration")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["PeMSD7", "METR-LA", "Chickenpox"],
    help="Select the dataset for training and evaluation"
)

# Encoder Type
encoder_type = st.sidebar.selectbox(
    "Encoder Type",
    ["Transformer", "GAT", "SAGE", "General"],
    help="Select the graph neural network encoder architecture"
)

# Training parameters
st.sidebar.header("Training Parameters")
epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=200, value=50, step=10,
                         help="Number of training epochs")
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f",
                           help="Learning rate for optimizer")
batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=256, value=64, step=16,
                             help="Batch size for training (Note: Currently using full sequence training)")
train_split = st.sidebar.slider("Training Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
                              help="Proportion of data used for training vs testing")

# Model architecture parameters
st.sidebar.header("Model Architecture")
hidden_channels = st.sidebar.slider("Hidden Channels", min_value=16, max_value=256, value=64, step=16,
                                  help="Number of hidden channels in the GNN encoder")
hidden_gru = st.sidebar.slider("GRU Hidden Size", min_value=16, max_value=256, value=64, step=16,
                             help="Hidden size for GRU decoder")
mlp_layers = st.sidebar.multiselect("MLP Decoder Layers", options=[16, 32, 64, 128, 256], default=[128, 64, 32],
                                  help="Hidden layers for MLP decoder")

# Optimizer options
optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"],
                                    help="Optimizer for training")
weight_decay = st.sidebar.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.0, step=0.0001, format="%.5f",
                                     help="L2 regularization strength")

# Add node selection to sidebar
st.sidebar.header("Visualization Options")
sample_node = st.sidebar.number_input("Node to visualize", min_value=0, value=0, step=1,
                                    help="Select which node to visualize in the predictions chart")

# --------------------------
# 1. Dataset Handling
# --------------------------
@st.cache_data
def load_dataset(dataset_name):
    with st.spinner(f"Loading {dataset_name} dataset..."):
        if dataset_name == 'PeMSD7':
            try:
                speed_data = pd.read_csv('dataset/PeMSD7_V_228.csv', header=None)
                adjacency_matrix = pd.read_csv('dataset/PeMSD7_W_228.csv', header=None).values
                edge_attr = None
                nodes = speed_data.shape[1]
                timesteps = speed_data.shape[0]
                st.info(f"PeMSD7 dataset loaded: {nodes} nodes, {timesteps} timesteps")
            except FileNotFoundError:
                st.error("PeMSD7 dataset files not found. Please ensure PeMSD7_V_228.csv and PeMSD7_W_228.csv are in the dataset folder.")
                return None, None, None, None, None

        elif dataset_name == 'METR-LA':
            try:
                with h5py.File('dataset/metr-la.h5', 'r') as f:
                    speed_data = pd.DataFrame(f['df']['block0_values'][:])
                with open('dataset/adj_mx.pkl', 'rb') as f:
                    adj_mx = pickle.load(f, encoding='latin1')
                adjacency_matrix = adj_mx[2]
                nodes = speed_data.shape[1]
                timesteps = speed_data.shape[0]
                st.info(f"METR-LA dataset loaded: {nodes} nodes, {timesteps} timesteps")
            except FileNotFoundError:
                st.error("METR-LA dataset files not found. Please ensure metr-la.h5 and adj_mx.pkl are in the dataset folder.")
                return None, None, None, None, None

        elif dataset_name == 'Chickenpox':
            try:
                speed_data = pd.read_csv('dataset/hungary_chickenpox.csv')
                adjacency_matrix = pd.read_csv('dataset/hungary_county_edges.csv').values
                edge_attr = None
                speed_data = speed_data.select_dtypes(include=[np.number])
                nodes = speed_data.shape[1]
                timesteps = speed_data.shape[0]
                st.info(f"Chickenpox dataset loaded: {nodes} nodes, {timesteps} timesteps")
            except FileNotFoundError:
                st.error("Chickenpox dataset files not found. Please ensure hungary_chickenpox.csv and hungary_county_edges.csv are in the dataset folder.")
                return None, None, None, None, None

        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(speed_data)

        if dataset_name == 'PeMSD7' or dataset_name == 'Chickenpox':
            edge_index = torch.tensor(np.stack(np.nonzero(adjacency_matrix)), dtype=torch.long)
            edge_attr = None
        else:
            adj_sparse = sp.coo_matrix(adjacency_matrix)
            edge_index = torch.tensor(np.vstack([adj_sparse.row, adj_sparse.col]), dtype=torch.long)
            edge_attr = torch.tensor(adj_sparse.data, dtype=torch.float32).unsqueeze(1)

        return normalized_data, edge_index, edge_attr, scaler, nodes

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
    def __init__(self, in_channels, out_channels, hidden_gru=64, mlp_layers=[128, 64, 32]):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_gru, batch_first=True)
        self.mlp = MLP(hidden_gru, mlp_layers, out_channels)

    def forward(self, z):
        gru_out, _ = self.gru(z)
        return self.mlp(gru_out.squeeze(1))

class GNNModel(nn.Module):
    def __init__(self, input_size, hidden_channels, out_channels, edge_index, edge_attr=None,
                 encoder_type='Transformer', hidden_gru=64, mlp_layers=[128, 64, 32]):
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

        self.decoder = GRUDecoder(hidden_channels, out_channels, hidden_gru, mlp_layers)

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
def train_evaluate(normalized_data, edge_index, edge_attr, scaler, sample_node=0,
                   encoder_type='Transformer', epochs=50, lr=0.001,
                   hidden_channels=64, train_split=0.8, hidden_gru=64,
                   mlp_layers=[128, 64, 32], optimizer_name='Adam', weight_decay=0.0):
    tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
    input_size = tensor_data.shape[1]

    train_size = int(len(tensor_data) * train_split)
    train_data, test_data = tensor_data[:train_size], tensor_data[train_size:]

    model = GNNModel(
        input_size=input_size,
        hidden_channels=hidden_channels,
        out_channels=input_size,
        edge_index=edge_index,
        edge_attr=edge_attr,
        encoder_type=encoder_type,
        hidden_gru=hidden_gru,
        mlp_layers=mlp_layers
    )

    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.MSELoss()

    train_losses = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    start_time = time.time()

    metrics_container = st.container()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data[:-1])
        loss = loss_fn(output, train_data[1:])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
        elapsed = time.time() - start_time
        estimated_total = elapsed / (epoch + 1) * epochs
        remaining = estimated_total - elapsed

        status_text.text(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Time elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            fig = px.line(
                x=list(range(1, len(train_losses) + 1)),
                y=train_losses,
                labels={'x': 'Epoch', 'y': 'Loss'},
                title='Training Loss',
                template="plotly_white"
            )
            loss_chart.plotly_chart(fig, use_container_width=True)

    with torch.no_grad():
        model.eval()
        test_output = model(test_data[:-1])
        test_pred = test_output.numpy()
        test_true = test_data[1:].numpy()

        if scaler:
            test_pred_original = scaler.inverse_transform(test_pred)
            test_true_original = scaler.inverse_transform(test_true)
        else:
            test_pred_original = test_pred
            test_true_original = test_true

        mse = mean_squared_error(test_true, test_pred)
        mae = mean_absolute_error(test_true, test_pred)
        rmse = np.sqrt(mse)
        epsilon = 1e-10
        mape = np.mean(np.abs((test_true - test_pred) / (np.abs(test_true) + epsilon))) * 100

        with metrics_container:
            st.header("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse:.4f}")
            col2.metric("MAE", f"{mae:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("MAPE (%)", f"{mape:.2f}")

            st.subheader(f"Predictions vs Actual for Node {sample_node}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=test_true_original[:, sample_node],
                mode='lines',
                name='Actual'
            ))
            fig.add_trace(go.Scatter(
                y=test_pred_original[:, sample_node],
                mode='lines',
                name='Prediction'
            ))
            fig.update_layout(
                title=f"Predictions vs Actual for Node {sample_node}",
                xaxis_title="Time Step",
                yaxis_title="Value",
                legend_title="Legend",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            error = np.abs(test_true - test_pred)
            max_display = 50
            if error.shape[0] > max_display or error.shape[1] > max_display:
                time_slice = slice(0, min(error.shape[0], max_display))
                node_slice = slice(0, min(error.shape[1], max_display))
                error_display = error[time_slice, node_slice]
                st.info(f"Displaying error heatmap for first {error_display.shape[0]} timesteps and {error_display.shape[1]} nodes.")
            else:
                error_display = error

            fig = px.imshow(
                error_display,
                labels=dict(x="Node", y="Time Step", color="Absolute Error"),
                title="Prediction Error Heatmap",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

    return model, {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'train_losses': train_losses
    }

# --------------------------
# 4. Main App Execution
# --------------------------
with st.spinner("Loading dataset..."):
    normalized_data, edge_index, edge_attr, scaler, num_nodes = load_dataset(dataset_name)

if num_nodes is not None:
    st.sidebar.write(f"Total nodes: {num_nodes}")
    if sample_node >= num_nodes:
        sample_node = num_nodes - 1
        st.sidebar.warning(f"Selected node was out of range. Adjusted to {sample_node}.")

if st.button("Train Model"):
    if normalized_data is not None and num_nodes is not None:
        st.subheader(f"Training AGNet with {encoder_type} encoder on {dataset_name} dataset")
        mlp_layers_int = [int(layer) for layer in mlp_layers]
        try:
            model, metrics = train_evaluate(
                normalized_data,
                edge_index,
                edge_attr,
                scaler,
                sample_node=sample_node,
                encoder_type=encoder_type,
                epochs=epochs,
                lr=lr,
                hidden_channels=hidden_channels,
                train_split=train_split,
                hidden_gru=hidden_gru,
                mlp_layers=mlp_layers_int,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay
            )
            if st.button("Save Model"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"agnet_{dataset_name}_{encoder_type}_{timestamp}.pt"
                torch.save(model.state_dict(), model_path)
                st.success(f"Model saved to {model_path}")
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.exception(e)
else:
    st.info("Click 'Train Model' to start training AGNet on the selected dataset and parameters.")

with st.expander("About AGNet Architecture"):
    st.markdown("""
    ## AGNet: Adaptive Graph Neural Network

    AGNet is a spatio-temporal graph neural network designed for capturing both spatial dependencies through graph convolutions and temporal dependencies through a GRU-based decoder.

    ### Architecture Components:

    1. **Graph Neural Network Encoder**
       - Captures spatial relationships between nodes in the graph
       - Options: Transformer, GAT (Graph Attention), GraphSAGE, General Convolution

    2. **Temporal GRU Decoder**
       - Processes the encoded graph representations to capture temporal dependencies
       - Uses GRU (Gated Recurrent Unit) followed by MLP layers

    3. **Prediction Head**
       - Multi-layer perceptron that maps GRU outputs to predictions

    ### Applications:

    - Traffic forecasting (PeMSD7, METR-LA datasets)
    - Disease propagation modeling (Chickenpox dataset)
    - Can be adapted for other spatio-temporal forecasting tasks
    """)

with st.expander("Dataset Information"):
    st.markdown("""
    ### PeMSD7
    Traffic speed measurements collected from loop detectors in the California Performance Measurement System (PeMS) District 7, covering Los Angeles area freeways.

    ### METR-LA
    Traffic speed data collected from loop detectors in Los Angeles County highway system. Contains measurements from 207 sensors over 4 months.

    ### Hungary Chickenpox
    Weekly chickenpox cases reported in 20 counties of Hungary from 2005 to 2015. The graph is constructed based on geographical adjacency between counties.
    """)
