import json
import sys

# load json
json_path = sys.argv[1]
model_params_dict = json.load(open(json_path, 'r'))

# metadata

print("modelname="+model_params_dict["Model_Name"])
print("modeltype="+model_params_dict["Model_Type"])

# model specific params
if model_params_dict["Model_Type"] == "determinisicpolynomial":
    print("degrees="+model_params_dict["Degrees"])