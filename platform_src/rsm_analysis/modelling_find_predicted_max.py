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

SA_Hyper_Params = {
    "Temperature": 5,
    "Cooling_Schedule": 0.95,
    "N_Proposals": 100,
    "particles" : 12000,
    "Deep_Search": 0,
    "replicates" : 3,
    "decimal_point_threshold" : 1
}


print()
print("Finding Predicted Optimum..")

max_y_samples = model.find_max_y(
    model_path = model_path,
    **SA_Hyper_Params
    )
print()
print("Predicted Optimium:")
print(max_y_samples)
