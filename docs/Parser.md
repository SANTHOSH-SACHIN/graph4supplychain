# Create Temporal Graph
This function creates a temporal graph by combining the data from the different timestamps.
It first fetches the timestamps, and then for each timestamp, it fetches the data and preprocesses it.
It then computes the demand values for each timestamp, and uses them to create a classification task.
The data is then converted into a hetero data format, which is a format used by the PyTorch Geometric library.
The function then sets the values for the hetero data, and sets the mask for the classification task.
Finally, it returns the temporal graph and the hetero data.

## Parameters
--------------
- regression : bool
    If True, the function will create a regression task instead of a classification task.
- multistep : bool
    If True, the function will create a multistep task instead of a single step task.
- out_steps : int
    The number of steps for the multistep task.
- task : str
    The type of task, either 'df' or 'bd'.
    - df: demand forecast
    - bd: bottleneck detection
- threshold : int
    Limit for Bottleneck Classification.
- q : str
    The selected quartile.

## Returns
-----------
temporal_graphs : dict
    A dictionary of the temporal graphs, where each key is a timestamp and the value is a tuple of the graph and the hetero data.
hetero_obj : dict
    A dictionary of the hetero data, where each key is a timestamp and the value is the hetero data.


# Create Classes
This function creates a dictionary of class ranges from the given demand values.
It takes in the number of classes and the demand values, and returns a dictionary of class ranges.

## Parameters
--------------
- num_classes : int
    The number of classes to create.
- demand_values : dict
    A dictionary of demand values where the keys are the node IDs and the values are the demand values.

## Returns
-----------
class_ranges : dict
    A dictionary of class ranges where the keys are the node IDs and the values are the class ranges.

# Validate Parser
- This function takes the testing data csvs of subsequent timestamps and uses it for testing and validation of the constructed models.
- Example can be found in data/test_1 and data/test_2
- CSVs can be generated using the simulation application.


# Sparsity Dict
- This function takes in a data dictionary and a timestamp, and returns a dictionary
with the part IDs as keys and a list of dictionaries as values. Each inner dictionary
contains the target facility IDs as keys and a list of quantities as values.

- The function calculates the total variation of each part by summing up the absolute
differences of the `quantity` values in each PARTSToFACILITY relationship between consecutive timestamps.

- The function then splits the parts into quartiles based on their total variation, and
assigns each part to a quartile based on which quartile it belongs to.

Parameters:
    data (dict): The data dictionary (Graph) containing the part IDs and their corresponding
        quantities and target facility IDs.
    timestamp (int): The current timestamp.

Returns:
    dict: A dictionary with the part IDs as keys and a list of dictionaries as values,
        where each inner dictionary contains the target facility IDs as keys and a list of
        quantities as values.