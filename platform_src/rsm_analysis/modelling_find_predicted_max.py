### import
import sys
import json
import pickle
import pandas as pd
from sub_scripts.modelling import generate_feature_matrix, LinearRegressionModel
import os  # Import os to handle directories

#import the project path 
project_path = sys.argv[1]
model_path = sys.argv[2]
feature_to_model = sys.argv[3]
modeltype = sys.argv[4]


# Create "prediction" directory if it doesn't exist
prediction_path = os.path.join(model_path, "prediction")
if not os.path.exists(prediction_path):
    os.makedirs(prediction_path)




# get model
with open(model_path + "/" + "fitted_model.pkl", 'rb') as file:
    model = pickle.load(file)

model_config_dict = json.load(open(model_path + "model_config.json", "r"))

SA_Hyper_Params = model_config_dict["Simulated_Annealing_Params"]

print()
print("Finding Predicted Optimum..")

max_y_samples, max_x_list, max_y_list = model.find_max_y(
    model_path = model_path,
    **SA_Hyper_Params,
    produce_plots = False
    )
print()
print("Predicted Optimium:")
print(max_y_samples)
