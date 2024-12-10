from utils.parser_st import TemporalHeterogeneousGraphParser
from gnn_models import *
from utils.utils import download_model_button
import os
from tempfile import NamedTemporaryFile
import streamlit as st
import torch
import plotly.graph_objects as go
import torch.nn.functional as F
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.nn as nn
from tempfile import NamedTemporaryFile


st.subheader("Graph Neural Network Training Interface")
# Model Configuration Section
with st.sidebar.expander("🎯 Task Configuration", expanded=True):
    task_type = st.radio(
        "Select Task Type",
        ["Classification", "Regression"],
        help="Choose whether to perform node classification or regression",
    )

    # Store task type in session state
    st.session_state.task_type = task_type
    num_layers = st.number_input(
        "Number of Layers",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of graph convolutional layers",
    )
    layer_config = {}
    for i in range(num_layers):
        encoder_type = st.sidebar.selectbox(
            "Select Layer Type",
            ["SAGEConv", "GATConv", "GeneralConv" , "TransformerConv"],
            key=f"layer_{i}",
        )
        st.session_state[f"encoder_type_{i}"] = encoder_type

        # Layer-specific parameters based on encoder type
        if encoder_type == "SAGEConv":
            normalize = st.checkbox(f"Normalize Layer {i+1}", value=True)
            aggr = st.selectbox(
                f"Aggregation Type Layer {i+1}", ["mean", "max", "add"]
            )
            root_weight = st.checkbox(f"Root Weight Layer {i+1}", value=True)
            project = st.checkbox(f"Project Layer {i+1}", value=False)
            bias = st.checkbox(f"Bias Layer {i+1}", value=True)
            layer_config[f"layer{i+1}"] = {
                "normalize": normalize,
                "aggr": aggr,
                "root_weight": root_weight,
                "project": project,
                "bias": bias,
            }


        elif encoder_type == "GATConv":
            heads = st.number_input(
                f"Number of Attention Heads Layer {i+1}", min_value=1, value=1
            )
            dropout = st.slider(
                f"Dropout Layer {i+1}", min_value=0.0, max_value=1.0, value=0.0
            )
            concat = st.checkbox(f"Concatenate Heads Layer {i+1}", value=True)
            negative_slope = st.number_input(
                f"Negative Slope Layer {i+1}", min_value=0.0, value=0.2
            )
            fill_value = st.selectbox(
                f"Fill Value Layer {i+1}", ["mean", "add", "min" , "max" , "mul"], index=0
            )
            bias = st.checkbox(f"Bias Layer {i+1}", value=True)
            residual = st.checkbox(f"Residual Layer {i+1}", value=True)
            layer_config[f"layer{i+1}"] = {
                "heads": heads,
                "dropout": dropout,
                "concat": concat,
                "negative_slope": negative_slope,
                "fill_value": fill_value,
                "bias": bias,
                "residual": residual,
            }

        elif encoder_type == "GeneralConv":
            aggr = st.selectbox(
                f"Aggregation Type Layer {i+1}", ["add", "mean", "max"]
            )
            skip_linear = st.checkbox(f"Skip Linear Layer {i+1}", value=False)
            directed_msg = st.checkbox(f"Directed Message Layer {i+1}", value=True)
            heads = st.number_input(
                f"Number of Heads Layer {i+1}", min_value=1, value=1
            )
            attention = st.checkbox(f"Use Attention Layer {i+1}", value=False)
            attention_type = st.selectbox(
                f"Attention Type Layer {i+1}",
                ["additive", "dot_product"],
                index=0,
            )
            l2_normalize = st.checkbox(f"L2 Normalize Layer {i+1}", value=False)
            bias = st.checkbox(f"Bias Layer {i+1}", value=True)
            layer_config[f"layer{i+1}"] = {
                "aggr": aggr,
                "skip_linear": skip_linear,
                "directed_msg": directed_msg,
                "heads": heads,
                "attention": attention,
                "attention_type": attention_type,
                "l2_normalize": l2_normalize,
                "bias": bias,
            }
        
        elif encoder_type == "TransformerConv":
            heads = st.number_input(
                f"Number of Heads Layer {i+1}", min_value=1, value=1
            )
            concat = st.checkbox(f"Concatenate Heads Layer {i+1}", value=True)
            beta = st.checkbox (f"Use Beta Layer {i+1}", value=False)
            dropout = st.slider(
                f"Dropout Layer {i+1}", min_value=0.0, max_value=1.0, value=0.0
            )
            root_weight = st.checkbox(f"Root Weight Layer {i+1}", value=True)
            bias   = st.checkbox(f"Bias Layer {i+1}", value=True)
            layer_config[f"layer{i+1}"] = {
                "heads": heads,
                "concat": concat,
                "beta": beta,
                "dropout": dropout,
                "root_weight": root_weight,
                "bias": bias,
            }
            
with st.sidebar.expander("⚙️ Training Parameters", expanded=True):
    hidden_channels = st.number_input(
        "Hidden Channels",
        min_value=8,
        max_value=256,
        value=64,
        help="Number of hidden channels in the graph neural network",
    )

    # Device for training Radio Button
    device = "cpu"
    task_forecast = st.sidebar.radio(
    "Select Task", ["df", "bd"]
    )


    num_epochs = st.number_input(
        "Number of Epochs",
        min_value=1,
        max_value=10000,
        value=50,
        help="Number of training epochs",
    )

    learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.1,
        value=0.001,
        format="%.4f",
        help="Learning rate for the optimizer",
    )

    patience = st.number_input(
        "Early Stopping Patience",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of epochs to wait before early stopping",
    )
local_dir = "./data"
# Data Configuration
with st.sidebar.expander("📊 Data Configuration", expanded=True):
    use_local_files = st.checkbox("Use Local Files", value=False)
    # metadata_path = ""

    local_dir = st.text_input("Local Directory Path", "./data")
    version = st.text_input(
        "Enter Version of the fetch",
        "NSS_1000_12_Simulation",
        key="train_graph_version",
    )
metadata_file = st.sidebar.file_uploader("Upload metadata.json", type="json")

if metadata_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(metadata_file.getvalue())
        temp_file_path = temp_file.name
# Start Training Button
    train_button = st.sidebar.button("🚀 Start Training")
    if train_button:
        try:
            if not use_local_files:
                try:
                    # Initialize parser and create temporal graph
                    base_url = os.getenv("SERVER_URL")
                    parser = TemporalHeterogeneousGraphParser(
                        base_url=base_url,
                        version=version,
                        headers={"accept": "application/json"},
                        meta_data_path=temp_file_path,
                        use_local_files=False,
                        local_dir=local_dir + "/",
                    )

                    # Create temporal graphs based on task type
                    regression_flag = (
                        True if st.session_state.task_type == "Regression" else False
                    )
                    
                    temporal_graphs, hetero_obj = parser.create_temporal_graph(
                        regression=regression_flag,
                        out_steps=3,
                        multistep=False,
                        task=task_forecast,
                        threshold=10,
                    )

                    # Select the graph (e.g., at time step 10)
                    G = temporal_graphs[len(temporal_graphs)][1]
                    # Initialize model
                    model = Model3(
                        hidden_channels=hidden_channels,
                        out_channels=G.num_classes,
                        max_num_parts=G.num_nodes,
                        G=G
                    )

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
                        (
                            trained_model,
                            best_test_accuracy,
                            best_test_loss,
                            epoch_losses,
                            epoch_accuracies,
                        ) = train_classification(
                            num_epochs,
                            model,
                            optimizer,
                            loss_fn,
                            temporal_graphs,
                            label="PARTS",
                            device=device,
                            patience=patience,
                        )
                        download_model_button(
                            trained_model, filename="SingleStepGNNClassification.pth"
                        )

                    else:
                        if task_forecast is "df":
                            (
                                trained_model,
                                best_test_r2,
                                best_test_adjusted_r2,
                                best_test_mae,
                                epoch_losses,
                                epoch_r2_scores,
                                epoch_adjusted_r2_scores
                            ) = train_regression(
                                num_epochs,
                                model,
                                optimizer,
                                loss_fn,
                                temporal_graphs,
                                label="PARTS",
                                device=device,
                                patience=patience,
                            )
                        else:
                            (
                                trained_model,
                                best_test_r2,
                                best_test_adjusted_r2,
                                best_test_mae,
                                epoch_losses,
                                epoch_r2_scores,
                                epoch_adjusted_r2_scores
                            ) = train_regression(
                                num_epochs,
                                model,
                                optimizer,
                                loss_fn,
                                temporal_graphs,
                                label="FACILITY",
                                device=device,
                                patience=patience,
                            )
                        download_model_button(
                            trained_model, filename="SingleStepGNNRegression.pth"
                        )

                    # Display results
                    if task_type == "Classification":
                        st.success(
                            f"Training Completed. Best Test Accuracy: {best_test_accuracy:.4f}, Best Test Loss: {best_test_loss:.4f}"
                        )
                    else:
                        st.success(
                            f"Training Completed. Best R² Score: {best_test_r2:.4f}, Best MAE: {best_test_mae:.4f}"
                        )

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
                        unsafe_allow_html=True,
                    )

                    # Create columns for metrics

                    if task_type == "Classification":
                        col1, col2 = st.columns(2)
                        col1.markdown(
                            f"<div class='metric-card'><h3>Accuracy</h3><p>{best_test_accuracy:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )
                        col2.markdown(
                            f"<div class='metric-card'><h3>Loss</h3><p>{best_test_loss:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )

                        # Plot performance graph for loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_losses))),
                                y=epoch_losses,
                                mode="lines+markers",
                                name="Loss",
                            )
                        )
                        fig_loss.update_layout(
                            title="Epoch-wise Loss", xaxis_title="Epoch", yaxis_title="Loss"
                        )
                        st.plotly_chart(fig_loss)

                        # Plot performance graph for accuracy
                        fig_accuracy = go.Figure()
                        fig_accuracy.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_accuracies))),
                                y=epoch_accuracies,
                                mode="lines+markers",
                                name="Accuracy",
                            )
                        )
                        fig_accuracy.update_layout(
                            title="Epoch-wise Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                        )
                        st.plotly_chart(fig_accuracy)
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.markdown(
                            f"<div class='metric-card'><h3>R² Score</h3><p>{best_test_r2:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )
                        col2.markdown(
                            f"<div class='metric-card'><h3>Adjusted R² Score</h3><p>{best_test_adjusted_r2:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )
                        col3.markdown(
                            f"<div class='metric-card'><h3>MAE</h3><p>{best_test_mae:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )

                        # Plot performance graph for loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_losses))),
                                y=epoch_losses,
                                mode="lines+markers",
                                name="Loss",
                            )
                        )
                        fig_loss.update_layout(
                            title="Epoch-wise Loss", xaxis_title="Epoch", yaxis_title="Loss"
                        )
                        st.plotly_chart(fig_loss)

                        # Plot performance graph for R² score
                        fig_r2 = go.Figure()
                        fig_r2.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_r2_scores))),
                                y=epoch_r2_scores,
                                mode="lines+markers",
                                name="R² Score",
                            )
                        )
                        fig_r2.update_layout(
                            title="Epoch-wise R² Score",
                            xaxis_title="Epoch",
                            yaxis_title="R² Score",
                        )
                        st.plotly_chart(fig_r2)

                        # Plot performance graph for Adj R² score
                        fig_adjr2 = go.Figure()
                        fig_adjr2.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_adjusted_r2_scores))),
                                y=epoch_adjusted_r2_scores,
                                mode="lines+markers",
                                name="Adjusted R² Score",
                            )
                        )
                        fig_adjr2.update_layout(
                            title="Epoch-wise Adjusted R² Score",
                            xaxis_title="Epoch",
                            yaxis_title="Adjusted R² Score",
                        )
                        st.plotly_chart(fig_adjr2)   
                
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")
                    print(str(e))
            else:
                try:
                    # Initialize parser and create temporal graph
                    base_url = os.getenv("SERVER_URL")
                    parser = TemporalHeterogeneousGraphParser(
                        base_url=base_url,
                        version=version,
                        headers={"accept": "application/json"},
                        meta_data_path=temp_file_path,
                        use_local_files=True,
                        local_dir=local_dir + "/",
                    )

                    # Create temporal graphs based on task type
                    regression_flag = (
                        True if st.session_state.task_type == "Regression" else False
                    )
                    
                    temporal_graphs, hetero_obj = parser.create_temporal_graph(
                        regression=regression_flag,
                        out_steps=3,
                        multistep=False,
                        task=task_forecast,
                        threshold=10,
                    )

                    # Select the graph (e.g., at time step 10)
                    G = temporal_graphs[len(temporal_graphs)][1]
                    # Initialize model
                    model = Model3(
                        hidden_channels=hidden_channels,
                        out_channels=G.num_classes,
                        max_num_parts=G.num_nodes,
                        G=G
                    )

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
                        (
                            trained_model,
                            best_test_accuracy,
                            best_test_loss,
                            epoch_losses,
                            epoch_accuracies,
                        ) = train_classification(
                            num_epochs,
                            model,
                            optimizer,
                            loss_fn,
                            temporal_graphs,
                            label="PARTS",
                            device=device,
                            patience=patience,
                        )
                        download_model_button(
                            trained_model, filename="SingleStepGNNClassification.pth"
                        )

                    else:
                        if task_forecast is "df":
                            (
                                trained_model,
                                best_test_r2,
                                best_test_adjusted_r2,
                                best_test_mae,
                                epoch_losses,
                                epoch_r2_scores,
                                epoch_adjusted_r2_scores
                            ) = train_regression(
                                num_epochs,
                                model,
                                optimizer,
                                loss_fn,
                                temporal_graphs,
                                label="PARTS",
                                device=device,
                                patience=patience,
                            )
                        else:
                            (
                                trained_model,
                                best_test_r2,
                                best_test_adjusted_r2,
                                best_test_mae,
                                epoch_losses,
                                epoch_r2_scores,
                                epoch_adjusted_r2_scores
                            ) = train_regression(
                                num_epochs,
                                model,
                                optimizer,
                                loss_fn,
                                temporal_graphs,
                                label="FACILITY",
                                device=device,
                                patience=patience,
                            )
                        download_model_button(
                            trained_model, filename="SingleStepGNNRegression.pth"
                        )

                    # Display results
                    if task_type == "Classification":
                        st.success(
                            f"Training Completed. Best Test Accuracy: {best_test_accuracy:.4f}, Best Test Loss: {best_test_loss:.4f}"
                        )
                    else:
                        st.success(
                            f"Training Completed. Best R² Score: {best_test_r2:.4f}, Best MAE: {best_test_mae:.4f}"
                        )

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
                        unsafe_allow_html=True,
                    )

                    # Create columns for metrics

                    if task_type == "Classification":
                        col1, col2 = st.columns(2)
                        col1.markdown(
                            f"<div class='metric-card'><h3>Accuracy</h3><p>{best_test_accuracy:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )
                        col2.markdown(
                            f"<div class='metric-card'><h3>Loss</h3><p>{best_test_loss:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )

                        # Plot performance graph for loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_losses))),
                                y=epoch_losses,
                                mode="lines+markers",
                                name="Loss",
                            )
                        )
                        fig_loss.update_layout(
                            title="Epoch-wise Loss", xaxis_title="Epoch", yaxis_title="Loss"
                        )
                        st.plotly_chart(fig_loss)

                        # Plot performance graph for accuracy
                        fig_accuracy = go.Figure()
                        fig_accuracy.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_accuracies))),
                                y=epoch_accuracies,
                                mode="lines+markers",
                                name="Accuracy",
                            )
                        )
                        fig_accuracy.update_layout(
                            title="Epoch-wise Accuracy",
                            xaxis_title="Epoch",
                            yaxis_title="Accuracy",
                        )
                        st.plotly_chart(fig_accuracy)
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.markdown(
                            f"<div class='metric-card'><h3>R² Score</h3><p>{best_test_r2:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )
                        col2.markdown(
                            f"<div class='metric-card'><h3>Adjusted R² Score</h3><p>{best_test_adjusted_r2:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )
                        col3.markdown(
                            f"<div class='metric-card'><h3>MAE</h3><p>{best_test_mae:.4f}</p></div>",
                            unsafe_allow_html=True,
                        )

                        # Plot performance graph for loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_losses))),
                                y=epoch_losses,
                                mode="lines+markers",
                                name="Loss",
                            )
                        )
                        fig_loss.update_layout(
                            title="Epoch-wise Loss", xaxis_title="Epoch", yaxis_title="Loss"
                        )
                        st.plotly_chart(fig_loss)

                        # Plot performance graph for R² score
                        fig_r2 = go.Figure()
                        fig_r2.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_r2_scores))),
                                y=epoch_r2_scores,
                                mode="lines+markers",
                                name="R² Score",
                            )
                        )
                        fig_r2.update_layout(
                            title="Epoch-wise R² Score",
                            xaxis_title="Epoch",
                            yaxis_title="R² Score",
                        )
                        st.plotly_chart(fig_r2)

                        # Plot performance graph for Adj R² score
                        fig_adjr2 = go.Figure()
                        fig_adjr2.add_trace(
                            go.Scatter(
                                x=list(range(len(epoch_adjusted_r2_scores))),
                                y=epoch_adjusted_r2_scores,
                                mode="lines+markers",
                                name="Adjusted R² Score",
                            )
                        )
                        fig_adjr2.update_layout(
                            title="Epoch-wise Adjusted R² Score",
                            xaxis_title="Epoch",
                            yaxis_title="Adjusted R² Score",
                        )
                        st.plotly_chart(fig_adjr2)   
                
                except Exception as e:
                    st.error(f"An error occurred during training: {str(e)}")
                    print(str(e))                
        except Exception as e:
                st.error(f"An error occurred during training: {str(e)}")
                print(str(e))

else:
    st.sidebar.warning("Please upload the metadata.json file")