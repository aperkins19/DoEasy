import pandas as pd
import numpy as np
import sys
import json

#import the project path 

chosen_dir = sys.argv[1]
model_path = sys.argv[2]

# import model_config 
model_config_dict = json.load(open(model_path + "/model_config.json", "r"))

# import design parameter dict 
design_parameters_dict = json.load(open(chosen_dir + "/design_parameters.json", "r"))
design_input_variables = list(design_parameters_dict["Variables"].keys())

# get the training data
tidy_data = pd.read_csv(chosen_dir + "/Datasets/tidy_dataset.csv")
tidy_data_training = tidy_data[tidy_data["DataPointType"] != "Validation"]



# import model_params_dict
model_params_dict = json.load(open(model_path + "model_params.json", "r"))
model_terms = model_params_dict["model_terms"]

# get trim the design input variables
model_input_variables = [var for var in design_input_variables if var in model_terms]

# initialise unique levels dict
unique_levels_dict = {}

# plotting slice element dict
data_slicing_values_for_plotting = {}

# extract levels
for variable in model_input_variables:
    
    #get the unique levels
    unique_levels_dict[variable] = sorted(tidy_data_training[variable].unique().tolist())

    # choose a level for slicing during data plotting.
    # get the left learning middle element if not equal
    if len(tidy_data_training[variable].unique().tolist()) % 2 == 0:

        index = (len(tidy_data_training[variable].unique().tolist()) // 2) + 1 # add 1 for python indexing

    else:
        index = (len(tidy_data_training[variable].unique().tolist()) // 2) # 1 not added here for left leaning index
    
    # Select the left-middle element
    element = tidy_data_training[variable].unique().tolist()[index]
    # append
    data_slicing_values_for_plotting[variable] = element




# append to the model params
model_config_dict["Unique_Levels"] = unique_levels_dict
model_config_dict["Data_Slicing_Values_For_Plotting"] = data_slicing_values_for_plotting


# save
with open(model_path + "/model_config.json", 'w') as json_file:
    json.dump(model_config_dict, json_file, indent=4)
