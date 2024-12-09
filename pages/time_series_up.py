import streamlit as st
from ts_models.arima_st import SingleStepARIMA, MultiStepARIMA
from ts_models.sarima_st import SingleStepSarimaAnalyzer, MultiStepSarimaAnalyzer
from ts_models.cnnlstm_st import SingleStepCNNLSTMAnalyzer
from ts_models.xgboost_st import XGBoostForecaster
from ts_models.prophet_st import SingleStepProphet, MultiStepProphet
from utils.parser_st import TemporalHeterogeneousGraphParser
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import os
import numpy as np
from ts_models.sarima_updated import sarima_demand_forecast,forecast_node_multistep_sarima
from tempfile import NamedTemporaryFile


#utils

class StreamlitTimeSeriesAnalyzer:
    def __init__(
        self, metadata_path: str = None, server_url: str = None, headers: Dict = None
    ):
        """Initialize all forecasting models"""
        self.metadata_path = "./metadata.json"
        self.server_url = server_url
        self.headers = {"accept": "application/json"}

        # Initialize models only when needed
        self.models = {}

    def initialize_models(self):
        """Initialize all forecasting models"""
        if not self.models:
            self.models = {
                "single_step_arima": SingleStepARIMA(self.metadata_path),
                "multi_step_arima": MultiStepARIMA(self.metadata_path),
                "single_step_sarima": SingleStepSarimaAnalyzer(self.metadata_path),
                "multi_step_sarima": MultiStepSarimaAnalyzer(self.metadata_path),
                "single_step_cnnlstm": SingleStepCNNLSTMAnalyzer(self.metadata_path),
                "single_step_xgboost": XGBoostForecaster(self.metadata_path),
                "multi_step_xgboost": XGBoostForecaster(self.metadata_path),
                "single_step_prophet": SingleStepProphet(self.metadata_path),
                "multi_step_prophet": MultiStepProphet(self.metadata_path),
            }


# Plotting functions (from time series code)
def plot_single_step_forecast(
    train_demand,
    test_demand,
    demand_forecast,
    model_type,
    node_id,
    demand_mape,
    lookback=None,
):
    """Helper function to plot single-step forecasts"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        train_demand.index,
        train_demand.values,
        label="Training Data",
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    ax.plot(
        test_demand.index,
        test_demand.values,
        label="Test Data",
        marker="o",
        linestyle="-",
        linewidth=2,
    )

    if model_type in ["CNN-LSTM", "XGBoost"]:
        lookback = lookback or 3
        forecast_index = test_demand.index[lookback:]
        if len(forecast_index) > len(demand_forecast):
            forecast_index = forecast_index[: len(demand_forecast)]
        elif len(forecast_index) < len(demand_forecast):
            demand_forecast = demand_forecast[: len(forecast_index)]
    else:
        forecast_index = test_demand.index

    ax.plot(
        forecast_index,
        demand_forecast,
        label=f"{model_type} Forecast",
        marker="s",
        linestyle="--",
        linewidth=2,
    )

    ax.set_title(
        f"Single-Step {model_type} Demand Forecast for Node {node_id}\nTest Set MAPE: {demand_mape:.2f}%"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_multi_step_forecast(
    train_demand,
    test_demand,
    forecasts,
    mapes,
    model_type,
    node_id,
    forecast_horizons,
    lookback=None,
):
    """Helper function to plot multi-step forecasts"""
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot training and test data
    ax.plot(
        train_demand.index,
        train_demand.values,
        label="Training Data",
        marker="o",
        linestyle="-",
        linewidth=2,
        color="blue",
    )
    ax.plot(
        test_demand.index,
        test_demand.values,
        label="Test Data",
        marker="o",
        linestyle="-",
        linewidth=2,
        color="green",
    )

    # Color map for different forecast horizons
    colors = plt.cm.rainbow(np.linspace(0, 1, len(forecast_horizons)))

    # Plot forecasts for each horizon
    for forecast, horizon, color in zip(forecasts, forecast_horizons, colors):
        # Adjust forecast index based on the model type and horizon
        if model_type == "XGBoost":
            lookback = lookback or 3
            start_index = train_demand.index[-1]
            forecast_index = pd.date_range(
                start=start_index, periods=len(forecast), freq=test_demand.index.freq
            )[1:]
        else:
            start_index = test_demand.index[horizon - 1]
            forecast_index = pd.date_range(
                start=start_index, periods=len(forecast), freq=test_demand.index.freq
            )

        # Truncate forecast if needed
        if len(forecast_index) > len(forecast):
            forecast_index = forecast_index[: len(forecast)]
        elif len(forecast_index) < len(forecast):
            forecast = forecast[: len(forecast_index)]

        # Plot the forecast
        ax.plot(
            forecast_index,
            forecast,
            label=f"{horizon}-Step Forecast (MAPE: {mapes[horizon]:.2f}%)",
            marker="s",
            linestyle="--",
            linewidth=2,
            color=color,
        )

    ax.set_title(f"Multi-Step {model_type} Demand Forecast for Node {node_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


st.header("Time Series Forecasting")

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source", ["Local Directory", "Server"]
)
version = st.sidebar.text_input(
    "Enter Version of the fetch", "NSS_1000_12_Simulation", key="ts_version"
)
local_dir = st.sidebar.text_input("Enter local directory path", "./data")

# File uploader for metadata.json
metadata_file = st.sidebar.file_uploader("Upload metadata.json", type="json")

if metadata_file is not None:
    # Create a temporary file and write the uploaded content to it
    with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file.write(metadata_file.getvalue())
        temp_file_path = temp_file.name

    if data_source == "Local Directory":
        st.sidebar.header("Local Directory Settings")

        try:
            base_url = os.getenv("SERVER_URL")
            parser = TemporalHeterogeneousGraphParser(
                base_url=base_url,
                version=version,
                headers={"accept": "application/json"},
                meta_data_path=temp_file_path,
                use_local_files=True,
                local_dir=local_dir + "/",
                num_classes=20,
            )
            st.sidebar.success("Successfully loaded local files!")
        except Exception as e:
            st.sidebar.error(f"Error loading local files: {str(e)}")

    else:  # Server
        st.sidebar.header("Server Settings")
        server_url = os.getenv("SERVER_URL")
        local_dir = st.sidebar.text_input(
            "Enter local directory path", "./data", key="local_dir"
        )
        if server_url:
            try:
                base_url = os.getenv("SERVER_URL")
                parser = TemporalHeterogeneousGraphParser(
                    base_url=base_url,
                    version=version,
                    headers={"accept": "application/json"},
                    meta_data_path=temp_file_path,
                    use_local_files=False,
                    local_dir=local_dir + "/",
                    num_classes=20,
                )
                st.sidebar.success("Successfully connected to server!")
            except Exception as e:
                st.sidebar.error(f"Error connecting to server: {str(e)}")
        else:
            st.sidebar.warning("Please enter a server URL")

    # Create temporal graph and get demand DataFrame
    try:
        with st.spinner("Loading and processing data..."):
            temporal_graphs, hetero_obj = parser.create_temporal_graph(
                regression=False,
                out_steps=3,
                multistep=False,
                task="df",
                threshold=10,
            )
            demand_df = parser.get_df()
            demand_df.index = pd.to_datetime(demand_df.index)

            # Initialize analyzer
            analyzer = StreamlitTimeSeriesAnalyzer(
                metadata_path=temp_file_path,
                server_url=server_url if data_source == "Server" else None,
                headers=(
                    {"accept": "application/json"}
                    if data_source == "Server"
                    else None
                ),
            )
            analyzer.initialize_models()
            analysis_type = st.selectbox("Select Forecasting Type", ["Single-Step", "Multi-Step"])

            node_id = st.selectbox("Select Node for Analysis", demand_df.columns)

            if analysis_type == "Single-Step":
                model_type = st.selectbox("Select a Timeseries Model",["ARIMA", "SARIMA", "CNN-LSTM", "XGBOOST","PROPHET"])
                
            else:
                model_type = st.selectbox("Select a Timeseries Model", ["ARIMA","SARIMA","XGBOOST"])

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
                train_demand, test_demand = analyzer.models[
                    "single_step_arima"
                ].train_test_split(demand_df[node_id])
                
                if analysis_type == "Single-Step":
                    st.subheader(f"Single-Step {model_type} Forecasting")

                    # Get forecasts based on model type
                    if model_type == "ARIMA":
                        demand_forecast, demand_mape = analyzer.models[
                            "single_step_arima"
                        ].forecast_node(train_demand, test_demand)
                        fig = plot_single_step_forecast(
                        train_demand,
                        test_demand,
                        demand_forecast,
                        model_type,
                        node_id,
                        demand_mape,
                        lookback,
                    )
                        st.pyplot(fig)
                        plt.close()

                        st.metric(f"{model_type} Demand Forecast MAPE", f"{demand_mape:.2f}%")
                        
                    elif model_type == "SARIMA":
                        demand_forecast,demand_mape = sarima_demand_forecast(train_demand,test_demand)
                        fig = plot_single_step_forecast(
                        train_demand,
                        test_demand,
                        demand_forecast,
                        model_type,
                        node_id,
                        demand_mape,
                        lookback,
                    )
                        st.pyplot(fig)
                        plt.close()

                        st.metric(f"{model_type} Demand Forecast MAPE", f"{demand_mape:.2f}%")
                    
                    elif model_type == "CNN-LSTM":
                        with st.spinner("Training CNN-LSTM model..."):
                            demand_forecast, demand_mape = analyzer.models[
                                "single_step_cnnlstm"
                            ].forecast_node_cnn_lstm(
                                train_demand,
                                test_demand,
                                node_id,
                                "demand",
                                lookback=lookback,
                                epochs=num_epochs,
                                batch_size=batch_size,
                            )
                            fig = plot_single_step_forecast(
                            train_demand,
                            test_demand,
                            demand_forecast,
                            model_type,
                            node_id,
                            demand_mape,
                            lookback,
                            )
                            st.pyplot(fig)
                            plt.close()

                            st.metric(f"{model_type} Demand Forecast MAPE", f"{demand_mape:.2f}%")
                            
                    elif model_type == "PROPHET":
                        with st.spinner("Training Prophet model..."):
                            demand_forecast, demand_mape = analyzer.models[
                                "single_step_prophet"
                            ].forecast_node(train_demand, test_demand)
                            
                            fig = plot_single_step_forecast(
                                train_demand,
                                test_demand,
                                demand_forecast,
                                model_type,
                                node_id,
                                demand_mape,
                                lookback,
                            )
                            st.pyplot(fig)
                            plt.close()

                            st.metric(f"{model_type} Demand Forecast MAPE", f"{demand_mape:.2f}%")
                    
                    else:  # XGBoost
                        with st.spinner("Training XGBoost model..."):
                            demand_forecast, demand_mape = analyzer.models[
                                "single_step_xgboost"
                            ].single_step_forecast(train_demand, test_demand)
                            
                            fig = plot_single_step_forecast(
                                train_demand,
                                test_demand,
                                demand_forecast,
                                model_type,
                                node_id,
                                demand_mape,
                                lookback,
                            )
                            st.pyplot(fig)
                            plt.close()

                            st.metric(f"{model_type} Demand Forecast MAPE", f"{demand_mape:.2f}%")
                            
                elif analysis_type == "Multi-Step":
                    st.subheader(f"Multi-Step {model_type} Forecasting")
                    
                    forecast_horizons = st.multiselect(
                            "Select Forecast Horizons (Steps)",
                            [1, 2, 3, 4, 5, 6],
                            default=[1],
                        )
                    
                    if forecast_horizons:
                            if model_type == "ARIMA":
                                forecasts, mapes = analyzer.models['multi_step_arima'].forecast_node_multistep(
                                    train_demand, test_demand, forecast_horizons
                                    )
                                fig = analyzer.models['multi_step_arima'].plot_multistep_forecast_comparison(
                                    train_series=train_demand, 
                                    test_series=test_demand, 
                                    forecasts=forecasts, 
                                    forecast_horizons=forecast_horizons,
                                    title=f"Multi-Step ARIMA Forecast for Node {node_id}",
                                    metric="Demand",
                                    node_id=node_id,
                                    mapes=mapes
                                )
                                
                                # Display the plot in Streamlit
                                st.pyplot(fig)
                                plt.close()
                                
                                # Display MAPE for each horizon
                                st.write("MAPE for Each Forecast Horizon:")
                                for horizon, mape in mapes.items():
                                    st.metric(f"{horizon}-Step Forecast MAPE", f"{mape:.2f}%")

                            elif model_type == "SARIMA":
                                forecasts, mapes = forecast_node_multistep_sarima(train_demand,test_demand,forecast_horizons)
                                fig = analyzer.models['multi_step_arima'].plot_multistep_forecast_comparison(
                                            train_series=train_demand, 
                                            test_series=test_demand, 
                                            forecasts=forecasts, 
                                            forecast_horizons=forecast_horizons,
                                            title=f"Multi-Step ARIMA Forecast for Node {node_id}",
                                            metric="Demand",
                                            node_id=node_id,
                                            mapes=mapes
                                        )
                                        
                                        # Display the plot in Streamlit
                                st.pyplot(fig)
                                plt.close()
                                        
                                        # Display MAPE for each horizon
                                st.write("MAPE for Each Forecast Horizon:")
                                for horizon, mape in mapes.items():
                                        st.metric(f"{horizon}-Step Forecast MAPE", f"{mape:.2f}%")
                            
                            else:  # XGBoost
                                with st.spinner("Training XGBoost model..."):
                                    forecasts, mapes = analyzer.models[
                                        "multi_step_xgboost"
                                    ].multi_step_forecast(
                                        train_demand, test_demand, forecast_horizons
                                    )

                                # Plot results
                                    fig = plot_multi_step_forecast(
                                        train_demand,
                                        test_demand,
                                        forecasts,
                                        mapes,
                                        model_type,
                                        node_id,
                                        forecast_horizons,
                                        lookback,
                                    )
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                col1, col2 = st.columns(2)
                                for i, horizon in enumerate(forecast_horizons):
                                    if i % 2 == 0:
                                        col1.metric(
                                            f"{horizon}-Step Forecast MAPE",
                                            f"{mapes[horizon]:.2f}%",
                                        )
                                    else:
                                        col2.metric(
                                            f"{horizon}-Step Forecast MAPE",
                                            f"{mapes[horizon]:.2f}%",
                                        )
                                        
            else:
                st.warning("Please select a node for analysis.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
else:
    st.sidebar.warning("Please upload the metadata.json file")

# st.subheader("Demand Data Preview")
# st.dataframe(demand_df)

