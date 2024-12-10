# Sparsity Page Documentation

The Sparsity page is designed to allow users to train a GNN model on a temporal heterogeneous graph. The page is divided into several sections, which are explained below. It is exactly similar to the Training of GNNs but the difference is that we are dividing the graph into quartiles and the user can select which quartile to use.

## Task Configuration

In this section, users can select the task type, which can be either **Single Step** or **Multi Step**. The task type determines the type of model to be trained.

### Single Step

If the task type is **Single Step**, the model will be trained to predict a single output value for each node in the graph.

### Multi Step

If the task type is **Multi Step**, the model will be trained to predict a sequence of output values for each node in the graph. The number of output steps can be specified by the user.

## Data Configuration

In this section, users can select the data configuration. The data configuration includes the following options:

### Use Local Files

If this option is selected, the app will use local files instead of fetching data from a server.

### Local Directory Path

The local directory path is the path to the directory containing the local files.

### Version

The version is the version of the data to be used.

### Quartile

The quartile is the quartile of the data to be used.

## Model Configuration

In this section, users can select the model configuration. The model configuration includes the following options:

### Hidden Channels

The hidden channels is the number of hidden channels in the model.

### Number of Layers

The number of layers is the number of layers in the model.

### Layer Type

The layer type is the type of layer to be used. The options are **SAGEConv**, **GATConv**, and **GeneralConv**.

### Normalization

The normalization option is available for **SAGEConv** and **GATConv** layers. If selected, the layer will be normalized.

### Attention

The attention option is available for **GATConv** layers. If selected, the layer will use attention.

### Aggregation

The aggregation option is available for **GeneralConv** layers. The options are **add**, **mean**, and **max**.

## Training Parameters

In this section, users can select the training parameters. The training parameters include the following options:

### Number of Epochs

The number of epochs is the number of epochs to train the model.

### Learning Rate

The learning rate is the learning rate for the optimizer.

### Early Stopping Patience

The early stopping patience is the number of epochs to wait before early stopping.

## Start Training

Once all the options are selected, users can start training the model by clicking the **Start Training** button.

## Model Performance Dashboard

After training is complete, the app will display a model performance dashboard. The dashboard will show the following metrics:

### Accuracy

The accuracy is the accuracy of the model on the test set.

### Loss

The loss is the loss of the model on the test set.

### R² Score

The R² score is the R² score of the model on the test set.

### MAE

The MAE is the mean absolute error of the model on the test set.

The dashboard will also show the epoch-wise performance of the model.

## Download Model

Once the model is trained, users can download the model by clicking the **Download Model** button. The model will be saved as a **.pth** file.
