"""
This script cleans and tidys the rawdata set and returns a csv ready for plotting.
It determines what type of plate reader etc to correctly process the data
"""
### import


import sys
import json
import pickle
import pandas as pd
from sub_scripts.modelling import generate_formula, generate_feature_matrix, LinearRegressionModel

#import the project path 
project_path = sys.argv[1]
model_path = sys.argv[2]
feature_to_model = sys.argv[3]
modeltype = sys.argv[4]


# import design parameters
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))

# import dataset
tidy_data = pd.read_csv(project_path+"/Datasets/tidy_dataset.csv")

# import coded design
coded_design = pd.read_csv(project_path+"/Experiment_Designs/design.csv")

# import model_params_dict 
model_params_dict = json.load(open(model_path + "model_params.json", "r"))
model_terms = model_params_dict["model_terms"]

# get variable name lists
design_input_variables = list(design_parameters_dict["Variables"].keys())
response_variables = list(design_parameters_dict["Response_Variables"].keys())


# generate coded design with response values
# first get the training data
training_data = tidy_data[tidy_data["DataPointType"] != "Validation"].copy()
input_matrix = training_data[design_input_variables].copy()
feature_matrix = generate_feature_matrix(input_matrix, model_terms)

# append Y
feature_matrix[feature_to_model] = tidy_data[feature_to_model]

# get trim the design input variables
model_input_variables = [var for var in design_input_variables if var in model_terms]



# # Merging coded_design with response variables from tidy_data via condition_id
# coded_design_with_response_values = coded_design.merge(tidy_data[["Condition_id"]+response_variables], on="Condition_id", how="right")
# # copy design
# numeric_coded_design_with_response_values = coded_design_with_response_values.copy()
# # replace all "-" and "+" to numeric equivialents 
# numeric_coded_design_with_response_values[input_variables] = numeric_coded_design_with_response_values[input_variables].replace({'-': -1, '+': 1})
# numeric_coded_design_with_response_values[input_variables] = numeric_coded_design_with_response_values[input_variables].astype(float)

#

# ## generate feature matrix
# input_matrix = numeric_coded_design_with_response_values[input_variables].copy()

# feature_matrix = generate_feature_matrix(input_matrix, model_terms)


# model_matrix = feature_matrix.copy()
# model_matrix[response_variables] = coded_design_with_response_values[response_variables]


model = LinearRegressionModel(
            feature_matrix = feature_matrix,
            model_terms = model_terms,
            input_variables = model_input_variables,
            response_variable=feature_to_model,
            design_parameters_dict = design_parameters_dict,
            model_path = model_path
            )

# fit
model.fit()

# save model
with open(model_path + "/" + "fitted_model.pkl", 'wb') as file:
    pickle.dump(model, file)





