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

# assess fit with training_data

training_results_df = model.assess_fit_with_training_data()


#################### if validation data present, conduct assessment
# import dataset
tidy_data = pd.read_csv(project_path+"/Datasets/tidy_dataset.csv")


# generate coded design with response values
# first get the validation data
if tidy_data['DataPointType'].str.contains('Validation').any():

    # import design parameters
    design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))

    # import coded design
    coded_design = pd.read_csv(project_path+"/Experiment_Designs/design.csv")

    # import model_params_dict 
    model_params_dict = json.load(open(model_path + "model_params.json", "r"))
    model_terms = model_params_dict["model_terms"]

    # get variable name lists
    input_variables = list(design_parameters_dict["Variables"].keys())
    response_variables = list(design_parameters_dict["Response_Variables"].keys())

    validation_data = tidy_data[tidy_data["DataPointType"] == "Validation"].copy()

    input_matrix = validation_data[input_variables].copy()
    feature_matrix = generate_feature_matrix(input_matrix, model_terms)

    # append Y
    feature_matrix[feature_to_model] = tidy_data[feature_to_model]

    validation_results_df = model.assess_fit_with_validation_data(feature_matrix)

    # Merge results
    training_results_df["Validation Data"] = validation_results_df["Validation Data"].copy()


print()
print("Model Fit Assessment: ")
print(training_results_df)
print()
training_results_df.to_csv(model_path + "Model_Fit_Assessment.csv", index=None)
print("Report saved in the model directort as Model_Fit_Assessment.csv")
print()


############# Observations vs Predictions

# build feature matrix containing both training and validation data if applicable

input_variables = list(design_parameters_dict["Variables"].keys())
model_terms = model_params_dict["model_terms"]

# regen feature matrix from tidydata
input_matrix = tidy_data[input_variables].copy()
feature_matrix = generate_feature_matrix(input_matrix, model_terms)


# append Y & Datapoint type
feature_matrix[feature_to_model] = tidy_data[feature_to_model]
feature_matrix["DataPointType"] = tidy_data["DataPointType"]

# generate obs vs preds plot
model.observations_vs_predictions(feature_matrix, project_path, model_path)


# save model to persist model stats
with open(model_path + "/" + "fitted_model.pkl", 'wb') as file:
    pickle.dump(model, file)
