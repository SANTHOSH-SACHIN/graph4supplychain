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

# Main function
def main():
    st.title("Supply Chain Forecasting & GNN Based Analysis")

# Run the main function
if __name__ == "__main__":
    main()
