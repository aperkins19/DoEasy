"""
This script cleans and tidys the rawdata set and returns a csv ready for plotting.
It determines what type of plate reader etc to correctly process the data
"""
### import

import sys
import json
import pickle
import pandas as pd
from sub_scripts.modelling import generate_feature_matrix, LinearRegressionModel
import os

#import the project path 
project_path = sys.argv[1]
select_model_dir = sys.argv[2]
feature_to_model = sys.argv[3]
select_model_name = sys.argv[4]
modeltype = sys.argv[5]

# import design parameters
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))

# import variable component names
X_names = list(design_parameters_dict["Variables"].keys())
Y_names = [feature_to_model]

# Load the prediction CSV file
file_path = select_model_dir + "prediction/input_prediction_dataset.csv"
prediction_data = pd.read_csv(file_path)

## input csv validation
# input variables match?
if not list(prediction_data.columns) == X_names:
    print("The columns in the uploaded input prediction csv file do not match the input variables used to train the model.")
    print("Please resolve by checking the docs.")
    sys.exit()

# import model 
# Loading the pickle file
model_pkl_path = select_model_dir + "/fitted_model.pkl"
with open(model_pkl_path, 'rb') as file:
    model = pickle.load(file)

# import model parameters
model_params_dict = json.load(open(select_model_dir + "model_params.json", 'r'))
model_terms = model_params_dict["model_terms"]

#import model config
model_config_dict = json.load(open(select_model_dir + "model_config.json", 'r'))

# prediction
# create input data
prediction_data[(feature_to_model + "_predicted")] = model.predict(generate_feature_matrix(prediction_data, model_terms))
prediction_data["Model"] = model_config_dict["Model_Name"]

prediction_data.to_csv(select_model_dir + "prediction/output_prediction_dataset.csv")