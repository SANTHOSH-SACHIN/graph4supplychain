from utils.parser_st import TemporalHeterogeneousGraphParser
from gnn_models import *
from ts_models.bottleneck_ts_model import POForecast
from utils.utils import generate_new_graph

import os
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

def get_bottleneck_nodes(demand_parts, facility_supply, id_map, threshold):
    pred = demand_parts['Predictions']
    d={}
    nodes=[]
    for idx, i in enumerate(pred):
        node_id = next((key for key, value in id_map.items() if value == idx), None)
        d[node_id] = i
    for i in facility_supply:
        if i in d:
            if d[i]/facility_supply[i] > threshold:
                nodes.append(i)
    return nodes

def visualize_bottleneck_nodes(nodes, parser, threshold):

    bottleneck_data = []
    for node in nodes:
        bottleneck_data.append({
            'Node ID': node,
            'Facility Type': 'LAM' if node in parser.lam_facility else 'External'
        })
    
    # Create DataFrame
    df_bottlenecks = pd.DataFrame(bottleneck_data)
    
    # Streamlit Visualization
    st.subheader("🚧 Bottleneck Nodes Analysis")
    
    # Metrics Display
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Bottleneck Nodes", value=len(nodes))
    with col2:
        st.metric(label="Bottleneck Percentage", 
                  value=f"{len(nodes) / len(parser.facility_supply) * 100:.2f}%")
    
    # Interactive Table
    st.dataframe(df_bottlenecks, 
                 column_config={
                     "Node ID": st.column_config.TextColumn("Node ID"),
                     "Facility Type": st.column_config.TextColumn("Facility Type")
                 },
                 hide_index=True)
    
    # Pie Chart of Facility Types
    if not df_bottlenecks.empty:
        type_counts = df_bottlenecks['Facility Type'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index, 
            values=type_counts.values,
            hole=.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title_text="Bottleneck Nodes by Facility Type",
            annotations=[dict(text='Facility Types', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional Insights
    st.markdown("### 🔍 Key Insights")
    st.markdown(f"""
    - **Total Bottleneck Nodes Detected**: {len(nodes)}
    - **Bottleneck Threshold**: {threshold}
    - **Facility Type Breakdown**: {', '.join([f"{count} {type}" for type, count in type_counts.items()])}
    """)


st.subheader("Graph Neural Network Bottleneck Detection")

# Data Configuration
with st.sidebar.expander("📊 Data Configuration", expanded=True):
    use_local_files = st.checkbox("Use Local Files", value=False)
    local_dir = st.text_input("Local Directory Path", "./data")
    version = st.text_input(
        "Enter Version of the fetch", "GNN_1000_12_v2", key="graphversion"
    )
    metadata_file = st.sidebar.file_uploader("Upload metadata.json", type="json")



try:
    # Initialize parser and create temporal graph
    base_url = os.getenv("SERVER_URL")
    if metadata_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(metadata_file.getvalue())
            temp_file_path = temp_file.name

        parser = TemporalHeterogeneousGraphParser(
            base_url=base_url,
            version=version,
            headers={"accept": "application/json"},
            meta_data_path=temp_file_path,
            use_local_files=use_local_files,
            local_dir=local_dir + "/",
        )

        temporal_graphs, hetero_obj = parser.create_temporal_graph(
            regression=True,
            out_steps=3,
            multistep=False,
            task="bd",
            threshold=10,
        )
        G = temporal_graphs[len(temporal_graphs)][1]
        st.sidebar.success('data loaded successfully')

except Exception as e:
    st.error(f"An error occurred during training: {str(e)}")
    print(str(e))


with st.sidebar.expander("Select model", expanded=True):
    model_type = st.write("Model type:", "Regression")
    step_type = st.write("Step type:", "singlestep")
    task_type = st.write("Task type: bd")

    model_file = st.file_uploader("Upload model (.pth):", type=["pth"])
    if model_file is not None:
        model_path = f"./{model_file.name}"
        with open(model_path, "wb") as f:
            f.write(model_file.getbuffer())
        try:
            model = torch.load(model_path)
            st.success(f"Model {model_file.name} loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    forecast_steps = st.number_input(
        "Enter timestamp",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Specify the number of future steps to forecast."
    )
    threshold = st.slider(
        label="Select Threshold", 
        min_value=0.0,            
        max_value=1.0,            
        value=0.5,                
        step=0.01                 
    )
st_button = st.sidebar.button("Run")

if st_button:
    po_df = parser.get_po_df()
    st.subheader("Product OfferingPreview")
    st.dataframe(po_df)

    forecaster = POForecast(po_df)
    dict_forecast = forecaster.function(forecast_steps)
    # st.write(dict_forecast)
    G = generate_new_graph(G, dict_forecast, parser.id_map['PRODUCT_OFFERING'])
    temporal_graphs = {1: ('placeholder', G)}
    demand_parts = test_single_step_regression(model, temporal_graphs, torch.nn.MSELoss(), label='FACILITY')
    nodes = get_bottleneck_nodes(demand_parts, parser.facility_supply, parser.id_map['FACILITY'], threshold)
    # st.write(nodes)
    visualize_bottleneck_nodes(nodes, parser, threshold)
    st.sidebar.success('Bottleneck detection completed successfully')
