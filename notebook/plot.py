import plotly.graph_objects as go
import pandas as pd

# Data for METR-LA Dataset
metr_la_data = {
    'Model': ['A3T-GCN', 'DCRNN'],
    'RMSE': [3.8989, 5.38],
    'MAE': [2.6480, 2.77],
    'MAPE': [168.376, 122.95],
}

# Data for PeMS-D7 Dataset
pems_d7_data = {
    'Model': ['ASTGCN', 'SSTGCN', 'ST-GCN'],
    'RMSE': [25.27, 59.7008, 6.90],
    'MAE': [16.63, 58.1919, 4.21],
    'MAPE': [192.87, 98.446, 160.32],
}

# Data for ChickenPox Dataset
chickenpox_data = {
    'Model': ['A3T-GCN', 'DCRNN', 'EvolveGCN-H', 'EvolveGCN-O', 'MPNN-LSTM', 'AGCRN'],
    'RMSE': [0.9752, 0.9672, 0.9786, 0.9777, 1.0679, 1.0209],
    'MAE': [0.6303, 0.6183, 0.6373, 0.6299, 0.7670, 0.6881],
    'MAPE': [542.53, 538.73, 944.008, 684.62, 2138.42, 822.22],
}

# Function to create and save plots
def create_and_save_plot(data, title, metrics, filename):
    df = pd.DataFrame(data)
    fig = go.Figure()

    # Add traces for each metric
    if 'RMSE' in metrics:
        fig.add_trace(go.Bar(x=df['Model'], y=df['RMSE'], name='RMSE', marker_color='rgb(34, 139, 34)'))  # Forest Green
    if 'MAE' in metrics:
        fig.add_trace(go.Bar(x=df['Model'], y=df['MAE'], name='MAE', marker_color='rgb(139, 0, 0)'))  # Dark Red
    if 'MAPE' in metrics:
        fig.add_trace(go.Bar(x=df['Model'], y=df['MAPE'], name='MAPE', marker_color='rgb(255, 165, 0)'))  # Orange

    fig.update_layout(
        title=title,
        xaxis_title='Model',
        yaxis_title='Error Metric',
        barmode='group',
        legend=dict(x=0.1, y=1.1, orientation="h"),
        template='plotly_white'
    )

    # Save the plot as an image
    fig.write_image(filename)

# Create and save plots for each dataset
create_and_save_plot(metr_la_data, 'RMSE and MAE on METR-LA Dataset', ['RMSE', 'MAE'], 'metr_la_rmse_mae.png')
create_and_save_plot(metr_la_data, 'MAPE on METR-LA Dataset', ['MAPE'], 'metr_la_mape.png')

create_and_save_plot(pems_d7_data, 'RMSE and MAE on PeMS-D7 Dataset', ['RMSE', 'MAE'], 'pems_d7_rmse_mae.png')
create_and_save_plot(pems_d7_data, 'MAPE on PeMS-D7 Dataset', ['MAPE'], 'pems_d7_mape.png')

create_and_save_plot(chickenpox_data, 'RMSE and MAE on ChickenPox Dataset', ['RMSE', 'MAE'], 'chickenpox_rmse_mae.png')
create_and_save_plot(chickenpox_data, 'MAPE on ChickenPox Dataset', ['MAPE'], 'chickenpox_mape.png')
