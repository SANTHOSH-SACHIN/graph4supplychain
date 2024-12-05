import io
import torch
import streamlit as st

def download_model_button(trained_model, filename="model.pth"):
    buffer = io.BytesIO()
    torch.save(trained_model, buffer)
    buffer.seek(0)
    st.download_button(
        label="Download Trained Model",
        data=buffer,
        file_name=filename,
        mime="application/octet-stream",
    )