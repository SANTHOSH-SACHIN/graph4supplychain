import streamlit as st
import torch

from utils.parser_st import TemporalHeterogeneousGraphParser
from gnn_models import (
    test_single_step_regression,
    test_single_step_classification,
    test_multistep_regression,
    test_multistep_classification,
)

st.write('Test Models')

#take local directory as input (path) in sidebar
with st.sidebar.expander("📊 Data Configuration", expanded=True):
    local_dir = st.text_input("Local Directory Path", "./data")
    version = st.text_input(
        "Enter Version",
        "test_1",
        key="test_graph_version",
    )

test_parser = TemporalHeterogeneousGraphParser(
            base_url='',
            version=version,
            headers={"accept": "application/json"},
            meta_data_path="metadata.json",
            use_local_files=True,
            local_dir=local_dir + "/",
        )

with st.sidebar.expander("Parse data", expanded=True):
    regression = st.selectbox("Regression task:", options=[True, False], index=1)
    multistep = st.selectbox("Multistep", options=[True, False], index=1)
    out_steps = 1 if not multistep else st.number_input("Output steps:", min_value=1, value=3)

    task = st.selectbox("Task type:", options=['df'], index=0)
    threshold = st.number_input("Threshold:", min_value=1, value=10)
    test_parser.validate_parser()
    st.success("CSV data converted to JSON format")
    temporal_graphs_test, hetero_obj_test = test_parser.create_temporal_graph(
        regression=regression,
        out_steps=out_steps,
        multistep=multistep,
        task=task,
        threshold=threshold
    )
    st.success("Test data parsed")

with st.sidebar.expander("Select model", expanded=True):
    model_type = st.write("Model type:", "Classification" if regression==False else "Regression")
    step_type = st.write("Step type:", "singlestep" if multistep==False else "multistep")
    task_type = st.write("Task type:", task)
    out_steps = st.write("Outsteps:", out_steps)
    model_file = st.file_uploader("Upload model (.pth):", type=["pth"])
    if model_file is not None:
        model_path = f"./{model_file.name}"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())

        try:
            model = torch.load(model_path)
            model.eval()  # Check if the model can switch to evaluation mode
            st.success(f"Model {model_file.name} loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {e}")

        task_type = "Classification" if regression==False else "Regression"
        step_type = "Single Step" if multistep==False else "Multistep"
        loss_fn = (
            torch.nn.MSELoss() if task_type == "Regression" else torch.nn.CrossEntropyLoss()
        )

        if step_type == "Single Step":
            temporal_data = temporal_graphs_test
            if task_type == "Regression":
                st.subheader("Testing Single-Step Regression Model")
                results = test_single_step_regression(model, temporal_data, loss_fn)
                st.write(results)
            else:
                st.subheader("Testing Single-Step Classification Model")
                results = test_single_step_classification(model, temporal_data, loss_fn)
                st.write(results)
        elif step_type == "Multistep":
            temporal_data = hetero_obj_test
            if task_type == "Regression":
                st.subheader("Testing Multistep Regression Model")
                results = test_multistep_regression(model, temporal_data, loss_fn)
                st.write(results)
            else:
                st.subheader("Testing Multistep Classification Model")
                results = test_multistep_classification(model, temporal_data, loss_fn)
                st.write(results)
