import streamlit as st
import pandas as pd
import numpy as np

# Dataset classification based on data volume
def classify_dataset(dataset_name):
    dataset_info = {
        "PEMS-D7": {
            "nodes": 228,
            "timespan": "6 months",
            "resolution": "5-minute intervals",
            "size_category": "Large dataset",
            "description": "California highway traffic data with 228 sensors"
        },
        "METR-LA": {
            "nodes": 207,
            "timespan": "4 months",
            "resolution": "5-minute intervals",
            "size_category": "Large dataset",
            "description": "Los Angeles traffic data collected from loop detectors"
        },
        "CHICKENPOX-HUNGARY": {
            "nodes": 20,
            "timespan": "10 years",
            "resolution": "Weekly",
            "size_category": "Medium dataset",
            "description": "Chickenpox cases across 20 counties in Hungary"
        },
        "Custom": {
            "nodes": "User defined",
            "timespan": "User defined",
            "resolution": "User defined",
            "size_category": "User defined",
            "description": "Custom dataset parameters"
        }
    }

    return dataset_info.get(dataset_name, dataset_info["Custom"])

# Define boundary conditions for selection parameters
def get_parameter_boundaries():
    boundaries = {
        "temporal_resolution": {
            "High (5-minute intervals)": "< 15 minutes between data points",
            "Medium (Hourly)": "15 minutes to 12 hours between data points",
            "Low (Weekly or longer)": "> 12 hours between data points"
        },
        "computational_resources": {
            "High (Powerful hardware available)": "GPU with 16+ GB VRAM, 32+ GB RAM",
            "Medium": "GPU with 8-16 GB VRAM, 16-32 GB RAM",
            "Limited": "CPU only or entry-level GPU, <16 GB RAM"
        },
        "prediction_horizon": {
            "Short-term": "1-24 time steps ahead",
            "Long-term": ">24 time steps ahead"
        },
        "data_volume": {
            "Large dataset": ">200 nodes or >100,000 time-node pairs",
            "Medium dataset": "50-200 nodes or 20,000-100,000 time-node pairs",
            "Sparse dataset": "<50 nodes or <20,000 time-node pairs"
        }
    }

    return boundaries

# Enhanced model recommendation system with structured framework
def get_model_recommendation(dataset_type, temporal_resolution, resources, prediction_horizon, data_volume, architecture_preference):
    # Scores Initialization
    models = {
        'DCRNN': 0,
        'A3TGCN': 0,
        'ASTGCN': 0,
        'SSTGCN': 0,
        'EvolveGCN': 0,
        'ST-GCN': 0
    }

    # Weight factors for different criteria
    weights = {
        'temporal': 1.5,  # Higher weight for temporal resolution
        'resources': 1.0,
        'prediction': 1.2,
        'data_volume': 1.0,
        'architecture': 0.8
    }

    # Apply dataset-specific adjustments
    if dataset_type != "Custom":
        dataset_info = classify_dataset(dataset_type)

        # Pre-set some parameters based on dataset characteristics
        if dataset_info["resolution"] == "5-minute intervals":
            models['DCRNN'] += 1 * weights['temporal']
            models['ST-GCN'] += 0.5 * weights['temporal']

        if dataset_info["size_category"] == "Large dataset":
            models['ASTGCN'] += 0.5 * weights['data_volume']
            models['SSTGCN'] += 0.5 * weights['data_volume']

        if dataset_info["resolution"] == "Weekly":
            models['EvolveGCN'] += 1 * weights['temporal']

    # Temporal Resolution scoring
    if temporal_resolution == "High (5-minute intervals)":
        models['DCRNN'] += 3 * weights['temporal']
        models['A3TGCN'] += 1 * weights['temporal']
        models['ST-GCN'] += 2 * weights['temporal']
    elif temporal_resolution == "Medium (Hourly)":
        models['A3TGCN'] += 2 * weights['temporal']
        models['ASTGCN'] += 2 * weights['temporal']
        models['ST-GCN'] += 2 * weights['temporal']
    else:  # Low
        models['A3TGCN'] += 3 * weights['temporal']
        models['EvolveGCN'] += 3 * weights['temporal']
        models['SSTGCN'] += 2 * weights['temporal']

    # Computational Resources scoring
    if resources == "High (Powerful hardware available)":
        models['DCRNN'] += 2 * weights['resources']
        models['ASTGCN'] += 3 * weights['resources']
    elif resources == "Medium":
        models['A3TGCN'] += 2 * weights['resources']
        models['ST-GCN'] += 3 * weights['resources']
    else:  # Limited
        models['SSTGCN'] += 3 * weights['resources']
        models['ST-GCN'] += 2 * weights['resources']

    # Prediction Horizon scoring
    if prediction_horizon == "Short-term":
        models['SSTGCN'] += 2 * weights['prediction']
        models['A3TGCN'] += 2 * weights['prediction']
    else:  # Long-term
        models['DCRNN'] += 2 * weights['prediction']
        models['ASTGCN'] += 2 * weights['prediction']
        models['EvolveGCN'] += 2 * weights['prediction']

    # Data Volume scoring
    if data_volume == "Large dataset":
        models['SSTGCN'] += 3 * weights['data_volume']
        models['ASTGCN'] += 2 * weights['data_volume']
        models['DCRNN'] += 2 * weights['data_volume']
    elif data_volume == "Medium dataset":
        models['A3TGCN'] += 2 * weights['data_volume']
        models['ST-GCN'] += 2 * weights['data_volume']
    else:  # Sparse dataset
        models['A3TGCN'] += 3 * weights['data_volume']
        models['EvolveGCN'] += 2 * weights['data_volume']

    # Architecture Preference scoring
    if architecture_preference == "Factorized (Faster training)":
        models['ST-GCN'] += 2 * weights['architecture']
        models['SSTGCN'] += 2 * weights['architecture']
    else:  # Coupled
        models['DCRNN'] += 2 * weights['architecture']
        models['ASTGCN'] += 2 * weights['architecture']

    # Normalize scores for better visualization
    max_score = max(models.values())
    if max_score > 0:
        for model in models:
            models[model] = round((models[model] / max_score) * 10, 1)  # Scale to 0-10

    # Get the model with highest score
    recommended_model = max(models.items(), key=lambda x: x[1])[0]

    return recommended_model, models

def main():
    st.title("STGNN Model Selection Advisor")
    st.write("""
    This tool helps you select the most appropriate Spatio-Temporal Graph Neural Network (STGNN)
    model based on your specific requirements and constraints.
    """)

    # Get dataset selection
    dataset_type = st.selectbox(
        "Select a predefined dataset or custom parameters:",
        ["Custom", "PEMS-D7", "METR-LA", "CHICKENPOX-HUNGARY"],
        help="Choose a known dataset or customize parameters"
    )

    # Display dataset info if selected
    if dataset_type != "Custom":
        dataset_info = classify_dataset(dataset_type)
        st.info(f"**{dataset_type} Dataset Information**")
        st.write(f"- Nodes: {dataset_info['nodes']}")
        st.write(f"- Timespan: {dataset_info['timespan']}")
        st.write(f"- Resolution: {dataset_info['resolution']}")
        st.write(f"- Size Category: {dataset_info['size_category']}")
        st.write(f"- Description: {dataset_info['description']}")

    st.subheader("Please answer the following questions:")

    # Get parameter boundaries for tooltips
    boundaries = get_parameter_boundaries()

    temporal_resolution = st.selectbox(
        "What is your required temporal resolution?",
        ["High (5-minute intervals)", "Medium (Hourly)", "Low (Weekly or longer)"],
        help=f"Select the time interval between data points:\n{boundaries['temporal_resolution']}"
    )

    resources = st.selectbox(
        "What computational resources do you have available?",
        ["High (Powerful hardware available)", "Medium", "Limited"],
        help=f"Consider your computing power and memory:\n{boundaries['computational_resources']}"
    )

    prediction_horizon = st.selectbox(
        "What is your prediction horizon requirement?",
        ["Short-term", "Long-term"],
        help=f"Prediction time steps:\n{boundaries['prediction_horizon']}"
    )

    data_volume = st.selectbox(
        "What is your data volume?",
        ["Large dataset", "Medium dataset", "Sparse dataset"],
        help=f"Consider nodes and temporal length:\n{boundaries['data_volume']}"
    )

    architecture_preference = st.selectbox(
        "What is your architecture preference?",
        ["Factorized (Faster training)", "Coupled (Better accuracy)"],
        help="Factorized architectures offer faster training, while coupled architectures provide better accuracy"
    )

    if st.button("Get Recommendation"):
        recommended_model, scores = get_model_recommendation(
            dataset_type, temporal_resolution, resources, prediction_horizon, data_volume, architecture_preference
        )

        st.success(f"### Recommended Model: {recommended_model}")

        # Display model characteristics
        st.subheader("Model Characteristics:")
        model_info = {
            'DCRNN': {
                'Computational Requirements': 'High',
                'Best Use-Case': 'High-frequency traffic data',
                'Key Strength': 'Superior performance with high-frequency data',
                'Architecture Type': 'Coupled',
                'Dataset Compatibility': 'PEMS-D7, METR-LA',
                'Training Epochs': '60-100',
                'Memory Requirement': 'High (16GB+ VRAM)'
            },
            'A3TGCN': {
                'Computational Requirements': 'Moderate',
                'Best Use-Case': 'Balanced applications',
                'Key Strength': 'Stability with data limitations',
                'Architecture Type': 'Hybrid',
                'Dataset Compatibility': 'All types',
                'Training Epochs': '30-50',
                'Memory Requirement': 'Medium (8GB+ VRAM)'
            },
            'ASTGCN': {
                'Computational Requirements': 'High',
                'Best Use-Case': 'Dense network analysis',
                'Key Strength': 'Rapid convergence (20 epochs)',
                'Architecture Type': 'Coupled',
                'Dataset Compatibility': 'PEMS-D7, METR-LA',
                'Training Epochs': '20-30',
                'Memory Requirement': 'High (16GB+ VRAM)'
            },
            'SSTGCN': {
                'Computational Requirements': 'Low',
                'Best Use-Case': 'Resource-constrained scenarios',
                'Key Strength': 'Efficient with large datasets',
                'Architecture Type': 'Factorized',
                'Dataset Compatibility': 'All types',
                'Training Epochs': '40-60',
                'Memory Requirement': 'Low (4GB+ VRAM)'
            },
            'EvolveGCN': {
                'Computational Requirements': 'Moderate',
                'Best Use-Case': 'Long-term temporal patterns',
                'Key Strength': 'Good performance with weekly intervals',
                'Architecture Type': 'Hybrid',
                'Dataset Compatibility': 'CHICKENPOX-HUNGARY',
                'Training Epochs': '50-80',
                'Memory Requirement': 'Medium (8GB+ VRAM)'
            },
            'ST-GCN': {
                'Computational Requirements': 'Moderate',
                'Best Use-Case': 'Baseline performance',
                'Key Strength': 'Reliable baseline with moderate compute',
                'Architecture Type': 'Factorized',
                'Dataset Compatibility': 'All types',
                'Training Epochs': '30-50',
                'Memory Requirement': 'Medium (8GB+ VRAM)'
            }
        }

        for key, value in model_info[recommended_model].items():
            st.write(f"**{key}:** {value}")

        # Display framework boundary conditions
        st.subheader("Parameter Boundary Conditions:")
        for param, conditions in boundaries.items():
            with st.expander(f"{param.title().replace('_', ' ')} Boundaries"):
                for level, description in conditions.items():
                    st.write(f"**{level}:** {description}")

        # Create scores dataframe
        st.subheader("Model Scoring Breakdown:")
        scores_df = pd.DataFrame(
            list(scores.items()),
            columns=['Model', 'Score']
        ).sort_values('Score', ascending=False)

        st.bar_chart(scores_df.set_index('Model'))

        # Display model comparison table
        st.subheader("STGNN Model Comparison:")
        comparison_data = {
            'Model': list(model_info.keys()),
            'Computational Req.': [model_info[m]['Computational Requirements'] for m in model_info],
            'Architecture': [model_info[m]['Architecture Type'] for m in model_info],
            'Memory': [model_info[m]['Memory Requirement'] for m in model_info],
            'Score': [scores.get(m, 0) for m in model_info]
        }

        comparison_df = pd.DataFrame(comparison_data).sort_values('Score', ascending=False)
        st.dataframe(comparison_df)
if __name__ == "__main__":
    main()
