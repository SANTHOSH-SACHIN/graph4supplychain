# Model Testing Dashboard


This page allows users to test their GNN models on a dataset of their choice. The user can upload their model and specify the dataset folder path containing timestamped CSV files, and the page will display the results of the model on the dataset.

## Configuration
--------------

The configuration section is located on the sidebar of the page. The user can enter the following information:

### Local Directory Path
The local directory path is the path to the folder where the user's dataset is stored. The user should enter the path to the folder that contains the dataset.

### Version
The version is the version of the dataset that the user wants to use. The user should enter the version number of the dataset.

### Metadata File
The metadata file is the file that contains the metadata of the dataset. The user should upload the metadata file.

## Model Selection
----------------

The user can upload the model by clicking on the "Upload Model" button. The model should be a .pth file.

Once the Model is uploaded , the testing configuration should be upldated in the Sidebar.Whether its Regression or Not and within that is it Single Step or Multi Step


## Results and Analysis
-------------------------

The results and analysis section is located on the main page. The page will display the results of the model on the dataset, including the metrics and the inferences.

### Metrics
The metrics are the results of the model on the dataset. The metrics are:

*   R-Squared Value (For Regression)
*   Adjusted R-Squared Value (For Regression)
*   Accuracy (For Classification)
*   Mean Absolute Error (MAE)

### Inferences
The inferences are the conclusions that can be drawn from the metrics. The inferences are:

*   If the R-Squared Value is less than 0.5, the model predictions are not very accurate.
*   If the R-Squared Value is between 0.5 and 0.8, the model has moderate predictive power.
*   If the R-Squared Value is greater than 0.8, the model has strong predictive power.
*   If the Accuracy is less than 0.5, the model performance is poor.
*   If the Accuracy is between 0.5 and 0.8, the model has acceptable performance.
*   If the Accuracy is greater than 0.8, the model is performing excellently.

## Uploading Model
----------------

The user can upload their model by clicking on the "Upload Model" button. The model should be a .pth file.


## Running the Model
-------------------

The user can run the model by clicking on the "Run Model" button. The page will display the results of the model on the dataset.

