import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import numpy as np
import pickle
import sys
import os
from sub_scripts.modelling import generate_feature_matrix, LinearRegressionModel



#import the project path
project_path = sys.argv[1]
model_path = sys.argv[2]
feature_to_model = sys.argv[3]
modeltype = sys.argv[4]


# import design
design = pd.read_csv(project_path + "/Experiment_Designs/design_real_values.csv")

# import the design_parameters_dict
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))
variables_dict = design_parameters_dict["Variables"]
design_input_variables = list(variables_dict.keys())


# get model
with open(model_path + "/" + "fitted_model.pkl", 'rb') as file:
    model = pickle.load(file)

# import plot_arguments 
model_config_dict = json.load(open(model_path + "model_config.json", "r"))
plot_args = model_config_dict["plot_args"]


# import model_params_dict
model_params_dict = json.load(open(model_path + "model_params.json", "r"))
model_terms = model_params_dict["model_terms"]


## first create the directory for the individual slice plots
dir_path = os.path.join(model_path, "individual_slices")

# Check if the directory exists and create it if it does not
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

# get trim the design input variables
model_input_variables = [var for var in design_input_variables if var in model_terms]


slice_datapoints_master = model_config_dict["Data_Slicing_Values_For_Plotting"].copy()


for variable in model_input_variables:

    data_slice_dict = slice_datapoints_master.copy()
    # Remove variable
    del data_slice_dict[variable]

    experiment_description = "varying_"+variable


    model.get_specific_slice_plots(
        data_slice_dict = data_slice_dict,
        experiment_description = experiment_description,
        model_path = model_path,
        project_path = project_path,
        plot_args = plot_args
        )


