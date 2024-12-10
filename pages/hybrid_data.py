from utils.parser_st import TemporalHeterogeneousGraphParser
import streamlit as st
import os
import pandas as pd
import ts_models.hybrid_st as hm

st.subheader("Hybrid Model")
# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source", ["Local Directory", "Server"]
)
version = st.sidebar.text_input(
    "Enter Version of the fetch", "NSS_1000_12_Simulation", key="hybridversion"
)
local_dir = "./data"
if data_source == "Local Directory":
    st.sidebar.header("Local Directory Settings")
    local_dir = st.sidebar.text_input("Enter local directory path", "./data/")
    base_url = os.getenv("SERVER_URL")
    try:

        parser = TemporalHeterogeneousGraphParser(
            base_url=base_url,
            version="",
            headers={"accept": "application/json"},
            meta_data_path="./metadata.json",
            use_local_files=True,
            local_dir=local_dir + "/" + version + "/",
            num_classes=20,
        )
        st.sidebar.success("Successfully loaded local files!")
    except Exception as e:
        st.sidebar.error(f"Error loading local files: {str(e)}")

else:  # Server
    st.sidebar.header("Server Settings")
    server_url = os.getenv("SERVER_URL")

    if server_url:
        version = st.sidebar.text_input(
            "Enter Version of the fetch", "NSS_1000_12_Simulation", key="version"
        )
        base_url = os.getenv("SERVER_URL")
        try:
            parser = TemporalHeterogeneousGraphParser(
                base_url=base_url,
                version=version,
                headers={"accept": "application/json"},
                meta_data_path="./metadata.json",
                use_local_files=False,
                local_dir=local_dir + "/" + version + "/",
                num_classes=20,
            )
            st.sidebar.success("Successfully connected to server!")

        except Exception as e:
            st.sidebar.error(f"Error connecting to server: {str(e)}")
    else:
        st.sidebar.warning("Please enter a server URL")
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
except Exception as e:
    st.error(f"Error processing data: {str(e)}")

model_choice = st.selectbox(
    "Select a Type", ["Non-Aggregated Columns", "Aggregated Columns"]
)
if model_choice == "Non-Aggregated Columns":
    part_id_list = []
    part_data = parser.get_extended_df()

    labels_df = parser.get_df()
    for x in labels_df.columns:
        part_data[x]["demand"] = labels_df[x]

    for i in labels_df.columns:
        part_id_list.append(i)

    node_id = st.selectbox("Select part id", labels_df.columns)
    if st.button("Run Forecasting"):
        viz, mape = hm.demand_forecasting(part_data, node_id)
        st.metric("Aggregated MAPE Scores:", mape)
        st.pyplot(viz)

elif model_choice == "Aggregated Columns":
    aggregation_method = st.radio(
        "Select Aggregation Method",
        ("mean", "sum", "median", "min", "max"),
        horizontal=True,
    )
    part_id_list = []
    labels_df = parser.get_df()
    node_id = st.selectbox("Select part id", labels_df.columns)
    part_data = parser.aggregate_part_features(node_id, aggregation_method)

    part_data["demand"] = labels_df[node_id]

    st.dataframe(part_data)

    for i in labels_df.columns:
        part_id_list.append(i)

    if st.button("Run Forecasting"):
        viz, mape = hm.aggregated_demand_forecasting(part_data, node_id)
        st.metric("Aggregated MAPE Scores:", mape)
        st.pyplot(viz)