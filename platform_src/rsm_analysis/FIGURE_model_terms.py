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


# import design
design = pd.read_csv(project_path + "/Experiment_Designs/design_real_values.csv")

# import the design_parameters_dict
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))
variables_dict = design_parameters_dict["Variables"]
variables_list = list(variables_dict.keys())


model_name = "Output/Protein_Yield/models/quadratic/"

model_path = project_path + model_name

# get model
with open(model_path + "/" + "fitted_model.pkl", 'rb') as file:
    model = pickle.load(file)


# import plot_arguments 
model_config_dict = json.load(open(model_path + "model_config.json", "r"))
plot_args = model_config_dict["plot_args"]


# get specific slice through mid contour


## first create the directory for the individual slice plots
dir_path = os.path.join(model_path, "model_term_assess")

# Check if the directory exists and create it if it does not
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

# data_slice_dict = {
#     "Pyruvate": 27.5,
#     "Phosphate": 27.5,
#     "Mg2+": 12.5
#     }

data_slice_dict = {
    "Phosphate": 27.5,
    "Mg2+": 12.5
    }

experiment_description = "model_terms"

model.model_term_significance_analysis(
    model_path,
    dont_plot_intercept=True,
    for_figure=True,
    tstat_absolute=True
    )