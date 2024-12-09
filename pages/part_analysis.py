from utils.parser_st import TemporalHeterogeneousGraphParser
import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta

def preprocess_dataframes(partsdf):
    """Convert tuples to datetime and create clean dataframes."""
    df_creation = partsdf.map(lambda x: x[0] if isinstance(x, tuple) else pd.NaT)
    df_expiry = partsdf.map(lambda x: x[1] if isinstance(x, tuple) else pd.NaT)
    return df_creation, df_expiry

def lifetime_analysis(df_creation, df_expiry, top_n=10):
    """Analyze part lifetimes with focus on top N longest-lived parts."""
    lifetimes = {}
    for col in df_creation.columns:
        valid_rows = df_creation[col].notna() & df_expiry[col].notna()
        part_lifetimes = (df_expiry[col][valid_rows] - df_creation[col][valid_rows]).dt.days
        lifetimes[col] = part_lifetimes.mean() if not part_lifetimes.empty else None
    
    # Sort and get top N parts by average lifetime
    top_parts = sorted(
        [(part, lifetime) for part, lifetime in lifetimes.items() if lifetime is not None], 
        key=lambda x: x[1], 
        reverse=True
    )[:top_n]
    
    return top_parts

def days_to_expire(df_creation, df_expiry, part_id, reference_date=None):
    """Calculate days to expiry for a specific part."""
    if reference_date is None:
        reference_date = datetime.now()
    
    else:
        reference_date = pd.Timestamp(reference_date)
    
    # Get most recent expiry date for the part
    expiry_dates = df_expiry[part_id].dropna()
    
    if expiry_dates.empty:
        return None
    
    most_recent_expiry = expiry_dates.max()
    
    # Two calculation methods
    days_from_current = (most_recent_expiry - reference_date).days
    days_from_creation = (most_recent_expiry - df_creation[part_id].max()).days
    
    return {
        'Days from Current Date': days_from_current,
        'Days from Creation Date': days_from_creation,
        'Most Recent Expiry': most_recent_expiry
    }

def predictive_maintenance_risk(df_creation, df_expiry, risk_threshold_days=30):
    """Assess maintenance risk across parts."""
    risk_analysis = {}
    current_date = datetime.now()
    
    for part_id in df_expiry.columns:
        expiry_dates = df_expiry[part_id].dropna()
        if not expiry_dates.empty:
            most_recent_expiry = expiry_dates.max()
            days_to_expiry = (most_recent_expiry - current_date).days
            
            # Risk categorization
            if days_to_expiry <= 0:
                risk_level = 'Expired'
            elif days_to_expiry <= risk_threshold_days:
                risk_level = 'High Risk'
            elif days_to_expiry <= 2 * risk_threshold_days:
                risk_level = 'Medium Risk'
            else:
                risk_level = 'Low Risk'
            
            risk_analysis[part_id] = {
                'Days to Expiry': days_to_expiry,
                'Risk Level': risk_level
            }
    
    return risk_analysis

def create_timeseries_plots(df_creation, df_expiry):
    """Create separate time series plots for part creation and expiry."""
    # Aggregate creation and expiry counts per timestamp
    creation_counts = df_creation.notna().sum(axis=1)
    expiry_counts = df_expiry.notna().sum(axis=1)
    
    return creation_counts, expiry_counts

def parts_status_on_date(df_creation, df_expiry, selected_date):
    """Get parts created and expired on a specific date."""
    selected_date = pd.Timestamp(selected_date)
    
    # Find parts created on the date
    created_parts = df_creation[df_creation == selected_date].count()
    
    # Find parts expired on the date
    expired_parts = df_expiry[df_expiry == selected_date].count()
    
    # Combine into a single dataframe
    status_df = pd.DataFrame({
        'Parts Created': created_parts,
        'Parts Expired': expired_parts
    }).fillna(0)
    
    return status_df




st.subheader("Parts Expiry Analysis")
local_dir = "./data"
# Data Configuration
base_url = os.getenv("SERVER_URL")
with st.sidebar.expander("📊 Data Configuration", expanded=True):
    use_local_files = st.checkbox("Use Local Files", value=True)

    if use_local_files:
        local_dir = st.text_input("Local Directory Path", "./data")

    base_url = os.getenv("SERVER_URL")
    version = st.sidebar.text_input(
        "Enter Version of the fetch", "GNN_1000_12_v2", key="graphversion"
    )
    headers = {"accept": "application/json"}

parser = TemporalHeterogeneousGraphParser(
    base_url=base_url,
    version=version,
    headers={"accept": "application/json"},
    meta_data_path="metadata.json",
    use_local_files=use_local_files,
    local_dir=local_dir + "/",
)
def parse_date(x):
    """Convert string dates to datetime objects."""
    if isinstance(x, tuple):
        # Assuming first and second elements are creation and expiry dates
        try:
            return (pd.to_datetime(x[0]), pd.to_datetime(x[1]))
        except Exception as e:
            return (pd.NaT, pd.NaT)
    return (pd.NaT, pd.NaT)
temporal_graphs, hetero_obj = parser.create_temporal_graph(regression=True)
partsdf = parser.get_date_df()
partsdf = partsdf.applymap(parse_date)

st.subheader("Parts Dates Preview")
st.dataframe(partsdf)

df_creation, df_expiry = preprocess_dataframes(partsdf)
analysis_type = st.selectbox('Choose Analysis', [
'Lifetime Analysis', 
'Days to Expire', 
'Predictive Maintenance Risk',
'Timeseries Creation/Expiry'
])

if analysis_type == 'Lifetime Analysis':
    top_n = st.slider('Top N Longest Lifetime Parts', 5, 20, 10)
    top_parts = lifetime_analysis(df_creation, df_expiry, top_n)
    
    st.subheader(f'Top {top_n} Parts by Average Lifetime')
    for part, lifetime in top_parts:
        st.metric(part, f'{lifetime:.2f} days')

elif analysis_type == 'Days to Expire':
    part_id = st.selectbox('Select Part ID', df_expiry.columns)
    reference_date = st.date_input('Reference Date', datetime.now())
    
    expiry_details = days_to_expire(df_creation, df_expiry, part_id, reference_date)
    
    if expiry_details:
        st.subheader(f'Expiry Details for {part_id}')
        for key, value in expiry_details.items():
            st.metric(str(key), str(value))

elif analysis_type == 'Predictive Maintenance Risk':
    risk_threshold = st.slider('Risk Threshold (Days)', 1, 90, 30)
    risk_analysis = predictive_maintenance_risk(df_creation, df_expiry, risk_threshold)
    
    st.subheader('Maintenance Risk Overview')
    risk_summary = {
        'Expired': sum(1 for r in risk_analysis.values() if r['Risk Level'] == 'Expired'),
        'High Risk': sum(1 for r in risk_analysis.values() if r['Risk Level'] == 'High Risk'),
        'Medium Risk': sum(1 for r in risk_analysis.values() if r['Risk Level'] == 'Medium Risk'),
        'Low Risk': sum(1 for r in risk_analysis.values() if r['Risk Level'] == 'Low Risk')
    }
    
    for risk, count in risk_summary.items():
        st.metric(risk, count)

elif analysis_type == 'Timeseries Creation/Expiry':
    creation_counts, expiry_counts = create_timeseries_plots(df_creation, df_expiry)
    
    # Create two subplots
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=creation_counts.index, y=creation_counts.values, 
                            mode='lines', name='Parts Created'))
    fig.add_trace(go.Scatter(x=expiry_counts.index, y=expiry_counts.values, 
                            mode='lines', name='Parts Expired'))
    fig.update_layout(title='Parts Creation and Expiry Over Time')
    
    st.plotly_chart(fig)
    
    # Date-specific parts status
    selected_date = st.date_input('Select Date for Parts Status')
    parts_status = parts_status_on_date(df_creation, df_expiry, selected_date)
    st.dataframe(parts_status)
