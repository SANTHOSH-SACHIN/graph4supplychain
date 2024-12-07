import streamlit as st
import pandas as pd
import os
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class SpatioTemporalModel:
    name: str
    datasets: List[str]
    results: Dict[str, Any]
    versions: List[str]
    markdown_file_path: str 
    image_folder_path: str

class SpatioTemporalModels:
    def __init__(self):
        self.models = {
            "STGCN": SpatioTemporalModel(
                name="Spatio-Temporal Graph Convolutional Networks",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {"MAE": 0.1983, "RMSE": 0.3601, "MSE": 0.1297, "r-squared": 0.8723}
                },
                versions=["v1.0", "v1.1", "v2.0"],
                markdown_file_path="./objects/st/stgcn.md",
                image_folder_path="./images/stgcn/"
            ),
            "DCRNN": SpatioTemporalModel(
                name="Diffusion Convolutional Recurrent Neural Network",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 0.9355, "MAE": 0.6183, "MAPE": 538.7384, "RMSE": 0.9672, "R-squared": 0.0709}
                },
                versions=["v1.0", "v2.0"],
                markdown_file_path="./objects/st/DCRNN.md",
                image_folder_path="./images/dcrnn/"
            ),
            "ASTGCN": SpatioTemporalModel(
                name="Attention-based Spatial-Temporal Graph Convolutional Networks",
                datasets=["PEMS04", "PEMS07"],
                results={
                    "PEMS04": {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"},
                    "PEMS07": {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
                },
                versions=["v1.0", "v1.1", "v2.0"],
                markdown_file_path="./objects/st/astgcn.md",
                image_folder_path="./images/astgcn/"
            ),
            "A3TGCN": SpatioTemporalModel(
                name="Adaptive Attention and Temporal Graph Convolutional Network",
                datasets=["METR-LA", "chickenpox"],
                results={
                    "METR-LA": {"MAE": 3.21, "RMSE": 0.4695, "MAPE": 168.1},
                    "chickenpox": {"MSE": 0.9509, "MAE": 0.6303, "MAPE": 542.5356, "RMSE": 0.9752, "R-squared": 0.0556}
                },
                versions=["v1.0", "v2.0"],
                markdown_file_path="./objects/st/a3tgcn.md",
                image_folder_path="./images/a3tgcn/"
            ),
            "SSTGNN": SpatioTemporalModel(
                name="Simplified Spatio-Temporal Traffic Forecasting Model Using Graph Neural Network",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {"MAE": 5.7507, "RMSE": 3.1506, "MAPE": 7.9441}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/sstgnn.md",
                image_folder_path="./images/sstgnn/"
            ),
            "STGAN": SpatioTemporalModel(
                name="Spatio-Temporal Generative Adversarial Network",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {"MAE": 3.848, "RMSE": 6.299, "MAPE": 9.210}
                },
                versions=["v1.0", "v2.0"],
                markdown_file_path="./objects/st/stgan.md",
                image_folder_path="./images/stgan/"
            ),
            "EMAGCN": SpatioTemporalModel(
                name="Exponential Moving Average Graph Convolutional Network",
                datasets=["PEMS07"],
                results={
                    "PEMS07": {"MAE": 0.3098, "RMSE": 0.5475, "MSE": 0.2998, "r-squared": 0.7048}
                },
                versions=["v1.0", "v2.0"],
                markdown_file_path="./objects/st/EMAGCN.md",
                image_folder_path="./images/emagcn/"
            ),
            # New Models for chickenpox dataset
            "AGCRN": SpatioTemporalModel(
                name="Attention-based Graph Convolutional Recurrent Network",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 1.0422, "MAE": 0.6881, "MAPE": 822.2222, "RMSE": 1.0209, "R-squared": -0.0364}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/agcrn.md",
                image_folder_path="./images/agcrn/"
            ),
            "DyGrEncoder": SpatioTemporalModel(
                name="Dynamic Graph Encoder",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 0.9655, "MAE": 0.6355, "MAPE": 1060.5135, "RMSE": 0.9826, "R-squared": 0.0411}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/dygrencoder.md",
                image_folder_path="./images/dygrencoder/"
            ),
            "EvolveGCNH": SpatioTemporalModel(
                name="Evolving Graph Convolutional Network (Heterogeneous)",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 0.9577, "MAE": 0.6373, "MAPE": 944.0087, "RMSE": 0.9786, "R-squared": 0.0488}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/evolvegcnh.md",
                image_folder_path="./images/evolvegcnh/"
            ),
            "EvolveGCNO": SpatioTemporalModel(
                name="Evolving Graph Convolutional Network (Original)",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 0.9558, "MAE": 0.6299, "MAPE": 684.6207, "RMSE": 0.9777, "R-squared": 0.0507}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/evolvegcno.md",
                image_folder_path="./images/evolvegcno/"
            ),
            "GCLSTM": SpatioTemporalModel(
                name="Graph Convolutional LSTM",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 1.1536, "MAE": 0.6915, "R-squared": -0.0925}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/gclstm.md",
                image_folder_path="./images/gclstm/"
            ),
            "GConvGRU": SpatioTemporalModel(
                name="Graph Convolutional GRU",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 0.9676, "MAE": 0.6274, "MAPE": 583.1867, "RMSE": 0.9837, "R-squared": 0.0390}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/gconvgru.md",
                image_folder_path="./images/gconvgru/"
            ),
            "GConvLSTM": SpatioTemporalModel(
                name="Graph Convolutional LSTM",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 0.9646, "MAE": 0.6288, "MAPE": 554.9057, "RMSE": 0.9821, "R-squared": 0.0420}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/gconvlstm.md",
                image_folder_path="./images/gconvlstm/"
            ),
            "LRGCN": SpatioTemporalModel(
                name="Low-Rank Graph Convolutional Network",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 1.0638}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/lrgcn.md",
                image_folder_path="./images/lrgcn/"
            ),
            "MPNNLSTM": SpatioTemporalModel(
                name="Message Passing Neural Network LSTM",
                datasets=["chickenpox"],
                results={
                    "chickenpox": {"MSE": 1.1404, "MAE": 0.7670, "MAPE": 2138.4280, "RMSE": 1.0679, "R-squared": -0.1326}
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/mpnnlstm.md",
                image_folder_path="./images/mpnnlstm/"
            ),
            "STGNN": SpatioTemporalModel(
                name="STGNN: Spatio-Temporal Graph Neural Networks",
                datasets=["METR-LA"],
                results={
                    "METR-LA": {
                        "15-minutes": {"MAE": 2.93, "MAPE": 7.54, "RMSE": 5.61},
                        "30-minutes":{"MAE": 3.29, "MAPE": 8.77, "RMSE": 6.47}, 
                        "60-minutes":{"MAE": 3.89, "MAPE": 11, "RMSE": 7.79}
                    }
                },
                versions=["v1.0"],
                markdown_file_path="./objects/st/stgnn.md",
                image_folder_path="./images/stgnn/"
            )

        }

    def display_images(self, image_folder_path: str):
        """Display all images in the specified folder."""
        try:
            image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                st.subheader("Dataset Images")
                for image_file in image_files:
                    image_path = os.path.join(image_folder_path, image_file)
                    st.image(image_path, caption=image_file, use_column_width=True)
            else:
                st.write("No images found in the folder.")
        except FileNotFoundError:
            st.write("Image folder not found. Please ensure the folder exists.")

    def display(self):
        st.title("Spatio-Temporal Graph Neural Networks")

        # Model selection with placeholder
        model_options = ["Select a model"] + list(self.models.keys())
        selected_model_name = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            key="model_selector"
        )

        if selected_model_name == "Select a model":
            st.info("Please select a model to proceed.")
            return

        model = self.models[selected_model_name]

        # Dataset and version selection
        col1, col2 = st.columns(2)
        with col1:
            dataset_options = ["Select a dataset"] + model.datasets
            selected_dataset = st.selectbox(
                "Select Dataset",
                dataset_options,
                index=0,
                key="dataset_selector"
            )
        with col2:
            version_options = ["Select a version"] + model.versions
            selected_version = st.selectbox(
                "Select Version",
                version_options,
                index=0,
                key="version_selector"
            )

        if selected_dataset == "Select a dataset" or selected_version == "Select a version":
            st.info("Please select both a dataset and a version to proceed.")
            return

        # Display basic information
        st.subheader("Model Information")
        st.markdown(f"**Model Name:** {model.name}")
        st.markdown(f"**Dataset:** {selected_dataset}")
        st.markdown(f"**Version:** {selected_version}")

        # Results for selected dataset
        st.subheader(f"Results for {selected_dataset}")
        if selected_dataset in model.results:
            results_for_dataset = model.results[selected_dataset]
            if isinstance(list(results_for_dataset.values())[0], dict):
                # Nested dictionary case
                results_df = pd.DataFrame.from_dict(results_for_dataset, orient='index')
                st.table(results_df)
            else:
                # Simple case
                results_df = pd.DataFrame.from_dict(
                    results_for_dataset,
                    orient='index',
                    columns=['Value']
                )
                st.table(results_df)
        else:
            st.write("No results available for the selected dataset.")

        # Display images
        image_folder = os.path.join(model.image_folder_path, selected_dataset)
        self.display_images(image_folder)
        st.write(f"Image folder path: {image_folder}")

        # Display markdown content
        try:
            with open(model.markdown_file_path, 'r') as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.write("Markdown file not found. Please ensure the file exists.")


def main():
    st.set_page_config(
        page_title="Spatio-Temporal GNN Models",
        page_icon="📊",
        layout="wide"
    )

    # Initialize the models and display the page content
    st_models = SpatioTemporalModels()
    st_models.display()

if __name__ == "__main__":
    main()
