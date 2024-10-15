import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import numpy as np

import sys


#import the project path 
project_path = sys.argv[1]


# import design
design = pd.read_csv(project_path + "/Experiment_Designs/design_real_values.csv")

# import the design_parameters_dict
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))
variables_dict = design_parameters_dict["Variables"]
variables_list = list(variables_dict.keys())
Y_variables_dict = design_parameters_dict["Response_Variables"]
Y_variables_list = list(Y_variables_dict.keys())

# append y variable columns
for y_variable in Y_variables_list:
    design[y_variable] = None


design.to_csv(project_path + "/Experiment_Designs/design_real_values.csv", index=False)
