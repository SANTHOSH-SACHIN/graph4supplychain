# import streamlit as st
# import torch

# from utils.parser_st import TemporalHeterogeneousGraphParser
# from gnn_models import (
#     test_single_step_regression,
#     test_single_step_classification,
#     test_multistep_regression,
#     test_multistep_classification,
# )

# st.write('Test Models')

# #take local directory as input (path) in sidebar
# with st.sidebar.expander("📊 Data Configuration", expanded=True):
#     local_dir = st.text_input("Local Directory Path", "./data")
#     version = st.text_input(
#         "Enter Version",
#         "test_1",
#         key="test_graph_version",
#     )

# test_parser = TemporalHeterogeneousGraphParser(
#             base_url='',
#             version=version,
#             headers={"accept": "application/json"},
#             meta_data_path="metadata.json",
#             use_local_files=True,
#             local_dir=local_dir + "/",
#         )

# with st.sidebar.expander("Parse data", expanded=True):
#     regression = st.selectbox("Regression task:", options=[True, False], index=1)
#     multistep = st.selectbox("Multistep", options=[True, False], index=1)
#     out_steps = 1 if not multistep else st.number_input("Output steps:", min_value=1, value=3)

#     task = st.selectbox("Task type:", options=['df'], index=0)
#     threshold = st.number_input("Threshold:", min_value=1, value=10)
#     test_parser.validate_parser()
#     st.success("CSV data converted to JSON format")
#     temporal_graphs_test, hetero_obj_test = test_parser.create_temporal_graph(
#         regression=regression,
#         out_steps=out_steps,
#         multistep=multistep,
#         task=task,
#         threshold=threshold
#     )
#     st.success("Test data parsed")

# with st.sidebar.expander("Select model", expanded=True):
#     model_type = st.write("Model type:", "Classification" if regression==False else "Regression")
#     step_type = st.write("Step type:", "singlestep" if multistep==False else "multistep")
#     task_type = st.write("Task type:", task)
#     out_steps = st.write("Outsteps:", out_steps)
#     model_file = st.file_uploader("Upload model (.pth):", type=["pth"])
#     if model_file is not None:
#         model_path = f"./{model_file.name}"
#         with open(model_path, "wb") as f:
#             f.write(model_file.getbuffer())

#         try:
#             model = torch.load(model_path)
#             model.eval()  # Check if the model can switch to evaluation mode
#             st.success(f"Model {model_file.name} loaded successfully")
#         except Exception as e:
#             st.error(f"Error loading model: {e}")

#         task_type = "Classification" if regression==False else "Regression"
#         step_type = "Single Step" if multistep==False else "Multistep"
#         loss_fn = (
#             torch.nn.MSELoss() if task_type == "Regression" else torch.nn.CrossEntropyLoss()
#         )

#         if step_type == "Single Step":
#             temporal_data = temporal_graphs_test
#             if task_type == "Regression":
#                 st.subheader("Testing Single-Step Regression Model")
#                 results = test_single_step_regression(model, temporal_data, loss_fn)
#                 st.write(results)
#             else:
#                 st.subheader("Testing Single-Step Classification Model")
#                 results = test_single_step_classification(model, temporal_data, loss_fn)
#                 st.write(results)
#         elif step_type == "Multistep":
#             temporal_data = hetero_obj_test
#             if task_type == "Regression":
#                 st.subheader("Testing Multistep Regression Model")
#                 results = test_multistep_regression(model, temporal_data, loss_fn)
#                 st.write(results)
#             else:
#                 st.subheader("Testing Multistep Classification Model")
#                 results = test_multistep_classification(model, temporal_data, loss_fn)
#                 st.write(results)

import streamlit as st
import torch
import pandas as pd
from utils.parser_st import TemporalHeterogeneousGraphParser
from gnn_models import (
    test_single_step_regression,
    test_single_step_classification,
    test_multistep_regression,
    test_multistep_classification,
)
from tempfile import NamedTemporaryFile
# Custom CSS for card styling
st.markdown("""
<style>
.card {
    background-color: #f0f2f5;
    border: 1px solid #d1d1d1;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-title {
    font-size: 20px;
    font-weight: bold;
    color: #333;
}
.metric-value {
    font-size: 24px;
    color: #007BFF;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title('Model Testing Dashboard')

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    local_dir = st.text_input("Local Directory Path", "./data")
    version = st.text_input("Enter Version", "test_1", key="test_graph_version")
    metadata_file = st.sidebar.file_uploader("Upload metadata.json", type="json")

    if metadata_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(metadata_file.getvalue())
            temp_file_path = temp_file.name

        test_parser = TemporalHeterogeneousGraphParser(
            base_url='',
            version=version,
            headers={"accept": "application/json"},
            meta_data_path=temp_file_path,
            use_local_files=True,
            local_dir=local_dir + "/",
        )

        regression = st.selectbox("Regression task:", options=[True, False], index=1)
        multistep = st.selectbox("Multistep", options=[True, False], index=1)
        out_steps = 1 if not multistep else st.number_input("Output steps:", min_value=1, value=3)
        task = st.selectbox("Task type:", options=['df'], index=0)
        threshold = st.number_input("Threshold:", min_value=1, value=10)

        if st.button("Parse Data"):
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

    # Main page for model selection and results
st.markdown("## 📊 Results and Analysis")

# File uploader for model
model_file = st.file_uploader("Upload Model (.pth)", type=["pth"])
if model_file:
    model_path = f"./{model_file.name}"
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())

    try:
        model = torch.load(model_path)
        model.eval()
        st.success(f"Model {model_file.name} loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    temporal_data = temporal_graphs_test if not multistep else hetero_obj_test

    # Model testing
    step_type = "Single Step" if not multistep else "Multistep"
    if step_type == "Single Step":
        if regression:
            # results is a dictionary containing the metrics , in that remove "Mae"
            results = test_single_step_regression(model, temporal_data, loss_fn)
        else:
            results = test_single_step_classification(model, temporal_data, loss_fn)
    else:
        if regression:
            results = test_multistep_regression(model, temporal_data, loss_fn)
        else:
            results = test_multistep_classification(model, temporal_data, loss_fn)


    # Analysis and Inferences
    st.markdown("### 🧠 Analysis and Inferences")
    if isinstance(results, dict):
        for metric_name, metric_value in results.items():
            # Generate inferences dynamically based on the metric
            if isinstance(metric_value, (int, float)):
                # For numeric metrics
                if metric_name.lower() == "r_squared":
                    st.metric("R-Squared Value", f"{metric_value:.2f}")
                    if metric_value < 0.5:
                        st.warning("⚠️ Model predictions are not very accurate.")
                    elif metric_value < 0.8:
                        st.info("🔍 Model has moderate predictive power.")
                    else:
                        st.success("✅ Model has strong predictive power.")
                if metric_name.lower() == "adjusted_r2":
                    st.metric("Adjusted R-Squared Value", f"{metric_value:.2f}")
                    if metric_value < 0.5:
                        st.warning("⚠️ Model predictions are not very accurate.")
                    elif metric_value < 0.8:
                        st.info("🔍 Model has moderate predictive power.")
                    else:
                        st.success("✅ Model has strong predictive power.")
                elif metric_name.lower() == "accuracy":
                    st.metric("Accuracy", f"{metric_value:.2%}")
                    if metric_value < 0.5:
                        st.error("❌ Model performance is poor.")
                    elif metric_value < 0.8:
                        st.warning("⚠️ Model has acceptable performance.")
                    else:
                        st.success("✅ Model is performing excellently.")
                else:
                    # General inference for other numeric metrics
                    st.metric(f"{metric_name.capitalize()}", f"{metric_value:.2f}")
                    if metric_value < 0.5:
                        st.warning(f"⚠️ {metric_name.capitalize()} indicates poor performance.")
                    elif metric_value < 0.8:
                        st.info(f"🔍 {metric_name.capitalize()} shows moderate performance.")
                    else:
                        st.success(f"✅ {metric_name.capitalize()} indicates excellent performance.")

            elif isinstance(metric_value, list):
                # Inferences for list metrics (e.g., trends or progressions)
                st.line_chart(pd.DataFrame(metric_value, columns=[metric_name.capitalize()]))
                st.info(f"📈 {metric_name.capitalize()} trend displayed above.")

            elif isinstance(metric_value, dict):
                # Inferences for dictionary metrics (e.g., category-specific performance)
                df_results = pd.DataFrame(metric_value.items(), columns=['Category', 'Value'])
                st.bar_chart(df_results.set_index('Category'))
                st.info(f"📊 {metric_name.capitalize()} breakdown displayed above.")
    else:
        st.error("No inferences could be drawn as results data is missing or invalid.")



