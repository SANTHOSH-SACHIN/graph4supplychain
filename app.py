import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv(dotenv_path='.env.default')

# Streamlit configuration
st.set_page_config(
    page_title="Supply Chain Forecasting & GNN Classification",
    page_icon="📈",
    layout="wide",
)

# Custom styles
def add_custom_styles():
    st.markdown(
        """
        <style>
        /* General page styles */
        body {
            font-family: "Arial", sans-serif;
            background-color: #f9f9f9;
        }
        .css-1d391kg p {
            font-size: 18px;
            color: #333;
        }

        /* Header style */
        .stTitle {
            font-size: 36px;
            font-weight: bold;
            color: #0066cc;
            text-align: center;
        }

        /* Navigation menu styles */
        .css-1v3fvcr {
            background-color: #e6f7ff;
            border-radius: 10px;
            padding: 10px;
        }

        /* Table styles */
        .dataframe {
            margin: 0 auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }

        /* Footer style */
        footer {
            font-size: 12px;
            text-align: center;
            color: #aaa;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main function
def main():
    add_custom_styles()  # Apply custom styles
    st.title("Supply Chain Forecasting & GNN Based Analysis")
    pages = {
        "GNN": [
            st.Page("pages/single_gnn.py", title="Single step"),
            st.Page("pages/multi_gnn.py", title="Multi step"),
            st.Page("pages/sparsity.py", title="Sparsity"),
        ],
        "Time Series": [
            st.Page("pages/time_series.py", title="time series"),
            st.Page("pages/time_series_up.py", title="time series up"),
            st.Page("pages/hybrid_data.py", title="Hybrid data"),
        ],
        "Analysis": [
            st.Page("pages/test_page.py", title="Test page"),
            st.Page("pages/bottleneck.py", title="Bottleneck"),
        ],
        "Benchmark": [
            st.Page("pages/spatio-models.py", title="Spatio models"),
            st.Page("pages/spatio-temporal-models.py", title="Spatio Temporal models"),
            st.Page("pages/complexity-analysis.py", title="Complexity Analysis"),
            st.Page("pages/comparison.py", title="Comparison"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()

# Run the main function
if __name__ == "__main__":
    main()
