import streamlit as st
import os
from PIL import Image

def create_app():
    # Set page config
    st.set_page_config(
        page_title="Complexity Analysis",
        layout="wide"
    )

    # Title
    st.title("Complexity Analysis")

    # Create columns for dropdowns
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox(
            "Select Model",
            ["None", "STGCN"],
            index=0  # Default to "None"
        )
    
    with col2:
        dataset = st.selectbox(
            "Select Dataset",
            ["None", "PEMS07"],
            index=0  # Default to "None"
        )

    # Only show content if both selections are made
    if model != "None" and dataset != "None":
        st.header(f"{model.upper()} Analysis on {dataset.upper()}")
        
        image_dir = f"images/{model.lower()}/{dataset}/complexity"
        
        try:
            # Display all images in the directory
            for image_file in sorted(os.listdir(image_dir)):
                if image_file.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, image_file)
                    image = Image.open(image_path)
                    
                    # Calculate new size (half of original)
                    new_size = (image.size[0] // 2, image.size[1] // 2)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Extract metric name from filename
                    metric_name = image_file.split('.')[0]
                    st.subheader(metric_name.capitalize())
                    st.image(image)  # Remove use_column_width=True to keep specific size
        except FileNotFoundError:
            st.error(f"Image directory not found: {image_dir}")
            st.info("Please ensure your images are stored in the correct directory structure.")
    else:
        st.info("Please select both a model and a dataset to view the analysis.")

if __name__ == "__main__":
    create_app()