import streamlit as st
import pandas as pd
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SpatialModel:
    name: str
    datasets: List[str]
    results: Dict[str, Dict[str, float]]
    versions: List[str]
    markdown_file_path: str 
    image_folder_path: str

class SpatialModels:
    def __init__(self):
        self.models = {
            "GCN": SpatialModel(
                name="Graph Convolutional Network",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 80.10, "F1-score": 0.7938, "Precision": 0.7789, "Recall": 0.8163}
                },
                versions=["v1.0", "v2.0"],
                markdown_file_path="./objects/spatial/GCN.md",
                image_folder_path="./images/gcn/"
            ),
            "GAT": SpatialModel(
                name="Graph Attention Network",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 82.9, "F1-score": 0.8260, "Precision": 0.8171, "Recall": 0.8260}
                },
                versions=["v1.0", "v1.1", "v2.0"],
                markdown_file_path="./objects/spatial/GAT.md",
                image_folder_path="./images/gat/"
            ),
            "GraphSAGE": SpatialModel(
                name="Graph Sample and Aggregate",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 80.6, "F1-score": 0.8069, "Precision": 0.8150, "Recall": 0.8060}
                },
                versions=["v1.0", "v1.2"],
                markdown_file_path="./objects/spatial/graphSAGE.md",
                image_folder_path="./images/graphsage/"
            ),
            "Neural": SpatialModel(
                name="Neural network",
                datasets=["CORA"],
                results={
                    "CORA": {"Accuracy": 49.60, "F1-score": 0.5039, "Precision": 0.5888, "Recall": 0.4960}
                },
                versions=["v1.0", "v1.2"],
                markdown_file_path="./objects/spatial/neural.md",
                image_folder_path="./images/neural/"
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
        st.title("Spatial Graph Neural Networks")

        # Model selection
        selected_model_name = st.selectbox(
            "Select Model",
            list(self.models.keys()),
            key="model_selector"
        )
        model = self.models[selected_model_name]

        # Dataset and version selection
        col1, col2 = st.columns(2)
        with col1:
            selected_dataset = st.selectbox(
                "Select Dataset",
                model.datasets,
                key="dataset_selector"
            )
        with col2:
            selected_version = st.selectbox(
                "Select Version",
                model.versions,
                key="version_selector"
            )

        # Display basic information
        st.subheader("Model Information")
        st.markdown(f"**Model Name:** {model.name}")
        st.markdown(f"**Dataset:** {selected_dataset}")
        st.markdown(f"**Version:** {selected_version}")

        # Results for selected dataset
        st.subheader(f"Results for {selected_dataset}")
        if selected_dataset in model.results:
            results_df = pd.DataFrame.from_dict(
                model.results[selected_dataset],
                orient='index',
                columns=['Value']
            )
            st.table(results_df)

        # Display images from the dataset
        self.display_images(model.image_folder_path+selected_dataset)
        st.write(model.image_folder_path+selected_dataset)

        # Display markdown file content if available
        try:
            with open(model.markdown_file_path, 'r') as f:
                st.markdown(f.read())
        except FileNotFoundError:
            st.write("Markdown file not found. Please ensure the file exists.")

def main():
    st.set_page_config(
        page_title="Spatial GNN Models",
        page_icon="📊",
        layout="wide"
    )

    # Initialize the models and display the page content
    spatial_models = SpatialModels()
    spatial_models.display()

if __name__ == "__main__":
    main()
