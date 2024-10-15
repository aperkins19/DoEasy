import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import numpy as np
import pickle
import sys

# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/app/platform_src/rsm_analysis/sub_scripts/')
 
from modelling import generate_feature_matrix, LinearRegressionModel

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

print(model)