# Graph Neural Network Training Interface
=============================================

This page is a graphical user interface for training a single graph neural network (GNN) on a dataset. The user can input various parameters and upload a metadata file to train the GNN.


## Layer Customization
------------

The user can customize the architecture of the GNN by selecting the number of layers, hidden channels, and dropout rate.

### Metadata File
The metadata file should be in JSON format and contain the complete information of the dataset according to the TemporalHeterogenousGraph class.

## Types of Layers
------------
 - SageConv
 - GATConv
 - GeneralConv
 - TransformerConv


## Parameters
------------

### Task Type
The user can select between "Classification" and "Regression" as the task type. This determines the type of loss function used during training.

 - Choose Classification if the task is demand calssification into two different percentile classes.
 - Choose Regression if the task is accurate demand forecasting.

### Number of Layers
The user can select the number of layers in the GNN. The default is 2.

### Hidden Channels
The user can select the number of hidden channels in the GNN. The default is 64.

### Dropout
The user can select the dropout rate for the GNN. The default is 0.1.

### Learning Rate
The user can select the learning rate for the optimizer. The default is 0.001.

### Number of Epochs
The user can select the number of epochs to train the GNN. The default is 50.

### Patience
The user can select the patience for early stopping. The default is 10.

### Metadata File
The user must upload a metadata file in JSON format. The metadata file should contain information about the dataset, such as the number of nodes, the number of features per node, and the labels for each node.


## Training
------------

When the user clicks the "Start Training" button, the app will train the GNN using the selected parameters and metadata file. The app will display the training loss and accuracy/MAE/R2 score at each epoch. The app will also display the final test accuracy/MAE/R2 score and download a file containing the trained model.
