import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class SpatioTemporalModel:
    name: str
    task: str
    datasets: List[str]
    results: Dict[str, Any]  # Flexible to handle different types

class ComparisonPage:
    def __init__(self):
        self.models = {
            "STGCN": SpatioTemporalModel(
                name="Spatio-Temporal Graph Convolutional Networks",
                task="Spatio-Temporal Forecasting",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {
                        "5-minutes": {"MAE": 0.1983, "RMSE": 0.3601, "MSE": 0.1297, "r-squared": 0.8723}
                    }
                }
            ),
            "DCRNN": SpatioTemporalModel(
                name="Diffusion Convolutional Recurrent Neural Network",
                task="Spatio-Temporal Forecasting",
                datasets=["METR-LA", "PeMS-BAY"],
                results={
                    "METR-LA": {
                        "15-minutes": {"MAE": 3.57, "RMSE": 7.12, "MAPE": 9.8}
                    },
                    "PeMS-BAY": {
                        "15-minutes": {"MAE": 2.95, "RMSE": 5.93, "MAPE": 7.8}
                    }
                }
            ),
            "ASTGCN": SpatioTemporalModel(
                name="Attention-based Spatial-Temporal Graph Convolutional Networks",
                task="Spatio-Temporal Forecasting",
                datasets=["PEMS04", "PEMS07"],
                results={
                    "PEMS04": {
                        "5-minutes": {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
                    },
                    "PEMS07": {
                        "5-minutes": {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
                    }
                }
            ),
            "SSTGNN": SpatioTemporalModel(
                name="Simplified Spatio-Temporal Traffic Forecasting Model Using Graph Neural Network",
                task="Spatio-Temporal Forecasting",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {
                        "45-minutes": {"MAE": 5.75, "RMSE": 3.15, "MAPE": 7.94}
                    }
                }
            ),
            "STGAN": SpatioTemporalModel(
                name="Spatio-Temporal Generative Adversarial Network",
                task="Spatio-Temporal Forecasting",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {
                          "45-minutes": {"MAE": 3.848, "RMSE": 6.299, "MAPE": 9.210}
                    },
                }
            ),
            "EMAGCN": SpatioTemporalModel(
                name="Exponential Moving Average Graph Convolutional Network",
                task="Spatio-Temporal Forecasting",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {
                        "5-minutes": {"MAE": 0.3098, "RMSE": 0.5475, "MSE": 0.2998, "r-squared": 0.7048}
                    }
                }
            ),
            "GCN": SpatioTemporalModel(
                name="Graph Convolutional Network",
                task="Node Classification",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 80.10, "F1-score": 0.7938, "Precision": 0.7789, "Recall": 0.8163}
                }
            ),
            "GAT": SpatioTemporalModel(
                name="Graph Attention Network",
                task="Node Classification",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 82.9, "F1-score": 0.8260, "Precision": 0.8171, "Recall": 0.8260}
                }
            ),
            "GraphSAGE": SpatioTemporalModel(
                name="Graph Sample and Aggregate",
                task="Node Classification",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 80.6, "F1-score": 0.8069, "Precision": 0.8150, "Recall": 0.8060}
                }
            ),
            "Neural": SpatioTemporalModel(
                name="Neural Network",
                task="Node Classification",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 49.60, "F1-score": 0.5039, "Precision": 0.5888, "Recall": 0.4960}
                }
            ),
            "A3TGCN": SpatioTemporalModel(
                name="Adaptive Attention and Temporal Graph Convolutional Network",
                task="Spatio-Temporal Forecasting",
                datasets=["METR-LA", "Chickenpox"],
                results={
                    "METR-LA": {
                        "15-minutes": {"MAE": 3.21, "RMSE": 0.4695, "MAPE": 168.1}
                    },
                    "Chickenpox": {
                        "1-week": {"MSE": 0.9509, "MAE": 0.6303, "MAPE": 542.5356, "RMSE": 0.9752, "R-squared": 0.0556}
                    }
                }
            ),
            "AGCRN": SpatioTemporalModel(
                name="Attention-based Graph Convolutional Recurrent Network",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 1.0422, "MAE": 0.6881, "MAPE": 822.2222, "RMSE": 1.0209, "R-squared": -0.0364}
                    }
                }
            ),
            "DyGrEncoder": SpatioTemporalModel(
                name="Dynamic Graph Encoder",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 0.9655, "MAE": 0.6355, "MAPE": 1060.5135, "RMSE": 0.9826, "R-squared": 0.0411}
                    }
                }
            ),
            "EvolveGCNH": SpatioTemporalModel(
                name="Evolving Graph Convolutional Network (Heterogeneous)",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 0.9577, "MAE": 0.6373, "MAPE": 944.0087, "RMSE": 0.9786, "R-squared": 0.0488}
                    }
                }
            ),
            "EvolveGCNO": SpatioTemporalModel(
                name="Evolving Graph Convolutional Network (Original)",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 0.9558, "MAE": 0.6299, "MAPE": 684.6207, "RMSE": 0.9777, "R-squared": 0.0507}
                    }
                }
            ),
            "GCLSTM": SpatioTemporalModel(
                name="Graph Convolutional LSTM",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 1.1536, "MAE": 0.6915, "R-squared": -0.0925}
                    }
                }
            ),
            "GConvGRU": SpatioTemporalModel(
                name="Graph Convolutional GRU",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 0.9676, "MAE": 0.6274, "MAPE": 583.1867, "RMSE": 0.9837, "R-squared": 0.0390}
                    }
                }
            ),
            "GConvLSTM": SpatioTemporalModel(
                name="Graph Convolutional LSTM",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 0.9646, "MAE": 0.6288, "MAPE": 554.9057, "RMSE": 0.9821, "R-squared": 0.0420}
                    }
                }
            ),
            "LRGCN": SpatioTemporalModel(
                name="Low-Rank Graph Convolutional Network",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 1.0638}
                    }
                }
            ),
            "MPNNLSTM": SpatioTemporalModel(
                name="Message Passing Neural Network LSTM",
                task="Spatio-Temporal Forecasting",
                datasets=["Chickenpox"],
                results={
                    "Chickenpox": {
                        "1-week": {"MSE": 1.1404, "MAE": 0.7670, "MAPE": 2138.4280, "RMSE": 1.0679, "R-squared": -0.1326}
                    }
                }
            ),
            "STGNN": SpatioTemporalModel(
                name="STGNN: Spatio-Temporal Graph Neural Networks",
                task="Spatio-Temporal Forecasting",
                datasets=["METR-LA"],
                results={
                    "METR-LA": {
                        "15-minutes": {"MAE": 2.93, "MAPE": 7.54, "RMSE": 5.61},
                        "30-minutes":{"MAE": 3.29, "MAPE": 8.77, "RMSE": 6.47}, 
                        "60-minutes":{"MAE": 3.89, "MAPE": 11, "RMSE": 7.79}
                    }
                },
            )
        }

        # Detailed dataset overview information
        self.dataset_info = {
            "PEMS07": (
                "PeMSD7 is traffic data in District 7 of California consisting of the traffic speed "
                "of 228 sensors while the period is from May to June in 2012 (only weekdays) with a "
                "time interval of 5 minutes. This dataset is popular for benchmarking traffic forecasting models."
            ),
            "METR-LA": (
                "A traffic forecasting dataset based on Los Angeles Metropolitan traffic conditions. "
                "The dataset contains traffic readings collected from 207 loop detectors on highways in "
                "Los Angeles County in aggregated 5-minute intervals for 4 months between March 2012 to June 2012."
            ),
            "PeMS-BAY": (
                "This traffic dataset is collected by California Transportation Agencies (CalTrans) Performance "
                "Measurement System (PeMS). It is represented by a network of 325 traffic sensors in the Bay Area with "
                "6 months of traffic readings ranging from Jan 1st, 2017, to May 31st, 2017, in 5-minute intervals."
            ),
            "CORA": (
                "The Cora dataset consists of 2708 scientific publications classified into one of seven classes. "
                "The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued "
                "word vector indicating the absence/presence of the corresponding word from the dictionary, which consists of 1433 unique words."
            ),
            "Chickenpox": (
                "A dataset of county-level chickenpox cases in Hungary between 2004 and 2014, made public during the "
                "development of PyTorch Geometric Temporal. The underlying graph is static, with vertices as counties and "
                "edges representing neighborhoods. Vertex features are lagged weekly counts of the chickenpox cases (including 4 lags). "
                "The target is the weekly number of cases for the upcoming week. The dataset consists of over 500 snapshots (weeks)."
            ),
        }

    def display_dataset_overview(self, dataset_name):
        """Displays an overview of the selected dataset."""
        st.subheader(f"Dataset Overview: {dataset_name}")
        st.write(self.dataset_info.get(dataset_name, "No information available for this dataset."))

    def display(self):
        st.title("Model Comparison")

        # Dropdowns for selecting task and dataset with placeholders
        tasks = ["Select Task"] + list(set([model.task for model in self.models.values()]))
        datasets = ["Select Dataset"] + list(set([dataset for model in self.models.values() for dataset in model.datasets]))

        selected_task = st.selectbox("Select Task", tasks, index=0)
        if selected_task == "Select Task":
            st.info("Please select a task to proceed.")
            return

        selected_dataset = st.selectbox("Select Dataset", datasets, index=0)
        if selected_dataset == "Select Dataset":
            st.info("Please select a dataset to proceed.")
            return

        self.display_dataset_overview(selected_dataset)

        # Collect available durations for the selected dataset among models
        durations = set()
        for model in self.models.values():
            if model.task == selected_task and selected_dataset in model.datasets:
                dataset_results = model.results.get(selected_dataset, {})
                if dataset_results:
                    # Check if results are nested (i.e., durations are present)
                    first_result = list(dataset_results.values())[0]
                    if isinstance(first_result, dict):
                        durations.update(dataset_results.keys())

        if durations:
            durations = ["Select Duration"] + sorted(durations)
            selected_duration = st.selectbox("Select Duration", durations, index=0)
            if selected_duration == "Select Duration":
                st.info("Please select a duration to proceed.")
                return
        else:
            selected_duration = None

        # Filter models based on the selected task, dataset, and duration
        filtered_models = []
        for model in self.models.values():
            if model.task == selected_task and selected_dataset in model.datasets:
                dataset_results = model.results.get(selected_dataset, {})
                if dataset_results:
                    first_result = list(dataset_results.values())[0]
                    if isinstance(first_result, dict):
                        # Nested results (with durations)
                        if selected_duration and selected_duration in dataset_results:
                            filtered_models.append(model)
                    else:
                        # Non-nested results (no durations)
                        if selected_duration is None:
                            filtered_models.append(model)

        # Display comparison results for the filtered models
        if filtered_models:
            st.subheader(f"Comparison of Models on {selected_dataset} ({selected_task})")

            # Gather all available metrics from the filtered models dynamically
            metrics = set()
            for model in filtered_models:
                dataset_results = model.results[selected_dataset]
                if selected_duration:
                    metrics.update(dataset_results[selected_duration].keys())
                else:
                    metrics.update(dataset_results.keys())

            # Prepare data for plotting
            comparison_data = {"Model": [model.name for model in filtered_models]}
            for metric in metrics:
                metric_values = []
                for model in filtered_models:
                    dataset_results = model.results[selected_dataset]
                    if selected_duration:
                        metric_value = dataset_results[selected_duration].get(metric, None)
                    else:
                        metric_value = dataset_results.get(metric, None)

                    # Convert metric_value to numeric if possible, else set to None
                    try:
                        metric_value = float(metric_value)
                    except (ValueError, TypeError):
                        metric_value = None

                    metric_values.append(metric_value)
                comparison_data[metric] = metric_values

            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)

            # Set Seaborn style
            sns.set_style("whitegrid")
            sns.set_palette("pastel")

            # Plot each metric
            for metric in metrics:
                # Extract data for the metric
                plot_data = comparison_df[['Model', metric]].dropna()

                if not plot_data.empty:
                    st.markdown(f"### {metric}")
                    # Create a bar chart using Seaborn
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(
                        x=metric,
                        y="Model",
                        data=plot_data,
                        ax=ax,
                        palette="viridis",
                        edgecolor=".6"
                    )
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Model')
                    ax.set_title(f'{metric} Comparison')
                    st.pyplot(fig)
                else:
                    st.write(f"No data available for metric: {metric}")
        else:
            st.write("No models available for the selected task, dataset, and duration.")


def main():
    st.subheader("Model Comparison")
    comparison_page = ComparisonPage()
    comparison_page.display()


main()
