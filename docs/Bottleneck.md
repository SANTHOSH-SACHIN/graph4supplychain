# Bottleneck Detection

This page is a part of the Graph Neural Network (GNN) demonstration application. It is designed to detect bottleneck nodes in a network using a GNN model. The page is user-friendly and does not require any data preparation or coding from the user.

## Overview

The bottleneck detection page is a user interface to detect nodes in a network that have a high demand compared to their capacity. The page allows users to select a model, input data, and set parameters for the detection process.

## How to Use

### Step 1: Data Configuration

Select if you want to use local data or server data. If you choose "Local Data", you can use the data available locally whose folder name can be given as version. If you choose "Server Data", you can select a dataset by entering the version name.

### Step 2: Upload Metadata File

Upload the metadata.json file associated with the Temporal Heterogeneous Graph. This file contains information about the nodes and edges in the graph.

### Step 3: Select Model
Upload the Trained Single Step Regression Model (.pth) associated with the **bd** task.

Set the parameters for the detection process. The parameters are:

* Threshold: The threshold value for detecting bottleneck nodes. A node is considered a bottleneck if its demand is greater than the threshold value.

The default value for the threshold is 0.5. To change the threshold value, enter a new value in the text box and click the "Apply" button.

### Step 4: Run the Detection

Click the "Run" button to start the detection process. The detection process will use the selected model and input data to detect bottleneck nodes in the network.

### Step 5: View the Results

The results of the detection process will be displayed in the main panel of the page. The results will include a list of bottleneck nodes, their demand and capacity, and a pie chart showing the distribution of bottleneck nodes by facility type.


### Common Issues

* The model is not selected: Make sure to upload a pre-trained model as .pth file.
* The data is not loaded: Make sure the data locally or in server is loaded.
* The threshold value is not set: Make sure to set the threshold value in the left sidebar. The default value is 0.5.
