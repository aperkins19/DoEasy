### import
import sys
import json
import pickle
import pandas as pd
from sub_scripts.modelling import generate_feature_matrix, LinearRegressionModel

#import the project path 
project_path = sys.argv[1]
model_path = sys.argv[2]
feature_to_model = sys.argv[3]
modeltype = sys.argv[4]


# get model
with open(model_path + "/" + "fitted_model.pkl", 'rb') as file:
    model = pickle.load(file)

# import model_params_dict 
model_params_dict = json.load(open(model_path + "model_params.json", "r"))
model_terms = model_params_dict["model_terms"]

# import design parameters
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))

# get variable name lists
input_variables = list(design_parameters_dict["Variables"].keys())
response_variables = list(design_parameters_dict["Response_Variables"].keys())

#################### if validation data present, conduct assessment
# import dataset
tidy_data = pd.read_csv(project_path+"/Datasets/tidy_dataset.csv")

# metadata
experiment_description = design_parameters_dict["Experiment_Name"]



# import plot_arguments 
model_config_dict = json.load(open(model_path + "model_config.json", "r"))
plot_args = model_config_dict["plot_args"]


# all individual plots
print()
print("Surface Plots")

model.get_all_surface_plots(
    experiment_description,
    model_path,
    project_path,
    plot_args = plot_args
    )
