"""
This script cleans and tidys the rawdata set and returns a csv ready for plotting.
It determines what type of plate reader etc to correctly process the data
"""
### import
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

from sub_scripts.response_surface_methadology_modelling import LinearModel, PolyModel
from sub_scripts.memory_management import determine_candidate_array_size_based_on_available_ram


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

# load in training dataset
tidy_data_path = project_path + "/Datasets/tidy_dataset.csv"
tidy_data = pd.read_csv(tidy_data_path)


# import model 
# Loading the pickle file
model_pkl_path = select_model_dir + "/" +select_model_name + "_fitted_model.pkl"
with open(model_pkl_path, 'rb') as file:
    model = pickle.load(file)

# import model parameters
model_params_dict = json.load(open(select_model_dir + "model_config.json", 'r'))

# create the model id string
model_id_string = model_params_dict["Model_Name"]


# predicted max y

if modeltype == "determinisiclinear":
    GB_free = 5
    # get the predicted max composition accorrding to the model.
    # candidate_array_size determined by dim size to prevent ram overflow
    candidate_array_size = determine_candidate_array_size_based_on_available_ram(
        X_names = X_names,
        data_type_size = 8,
        # RAM to leave available
        GB_free = GB_free
    )

elif modeltype == "determinisicpolynomial":
    GB_free = 10

    # get the predicted max composition accorrding to the model.
    # candidate_array_size determined by dim size to prevent ram overflow
    candidate_array_size = determine_candidate_array_size_based_on_available_ram(
        X_names = X_names,
        data_type_size = 8,
        # RAM to leave available
        GB_free = GB_free
    )




predicted_max_Y_summary, predicted_dataset_df = model.find_max_y(
    candidate_array_size = candidate_array_size
    )
predicted_max_Y_summary = predicted_max_Y_summary.to_frame().T
# round
predicted_max_Y_summary = predicted_max_Y_summary.round(1)
# add metadata
predicted_max_Y_summary["DataPointType"] = "PredictedMax"
predicted_max_Y_summary["ObservedResponse"] = "ToBeCollected"

# get the max y in the real data
max_yield = tidy_data[feature_to_model].max()

# Input for that max
max_input = tidy_data[tidy_data[feature_to_model]==max_yield][X_names].drop_duplicates()

# build summary
real_max_Y_summary = max_input.copy()
real_max_Y_summary["ObservedResponse"] = max_yield

# what the model predicts that input should be.
real_max_Y_summary["PredictedResponse"] = model.predict(max_input)
real_max_Y_summary["DataPointType"] = "ObservedMax"


## comparison to centerpoints

## generate mean and se of centerpoint Ys
# get centerpoint data summary
core_attributes = X_names + Y_names

centerpointdata = tidy_data[tidy_data["DataPointType"]=="Center"][core_attributes].drop_duplicates()
centerpoint_summary = centerpointdata[X_names].drop_duplicates()
centerpoint_summary["ObservedResponse"] = centerpointdata[Y_names].mean()[0]
#centerpoint_summary["ObservedResponseSEM"] = centerpointdata[Y_names].sem()[0]

centerpoint_summary["PredictedResponse"] = model.predict(centerpoint_summary[X_names])
centerpoint_summary["DataPointType"] = "ObservedCenter"
centerpoint_summary.reset_index(drop=True, inplace=True)


# report
report = pd.concat([real_max_Y_summary, centerpoint_summary, predicted_max_Y_summary])
report.to_csv(select_model_dir + "/prediction_report.csv")
print(report)


# Contour plots & Surface plots

experiment_description = model_id_string


if (modeltype == "determinisiclinear") or (modeltype == "determinisicpolynomial"):

    model.get_surface_plots_as_figure(
        experiment_description = experiment_description,
        plot_ground_truth=model_params_dict["plot_ground_truth"],
        modelpath=select_model_dir,
        feature_to_model = feature_to_model
        )

    model.get_contour_plots_as_figure(
        experiment_description = experiment_description,
        modelpath = select_model_dir,
        feature_to_model = feature_to_model,
        plot_ground_truth=model_params_dict["plot_ground_truth"]
        )


