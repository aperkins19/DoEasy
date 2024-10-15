import json
import pandas as pd
import math
import numpy as np
import sys
from auxillary_scripts.rsm_generation import *

#############################################################

#import the project path 
project_path = sys.argv[1]

#############################################################



# import the design_parameters_dict
design_parameters_path = project_path + "/design_parameters.json"
design_parameters_dict = json.load(open(design_parameters_path, 'r'))
variables_dict = design_parameters_dict["Variables"]

print(variables_dict)

variable_names = list(design_parameters_dict["Variables"].keys())
num_replicates = design_parameters_dict["Technical_Replicates"]


## generate design
if design_parameters_dict["Design_Type"] == "CCD":

    if design_parameters_dict.get("Specified_Alpha") is not None:
        alpha = design_parameters_dict["Specified_Alpha"]
    else:
        alpha = math.pow(math.pow(2,len(variables_dict)),0.25) # (2^k)**0.25


    design = CentralCompositeDesign(
        num_dimensions = len(variables_dict),
        alpha = alpha,
        num_center_points = 2,
        variable_names = variable_names,
        num_replicates = num_replicates
        )
    print()
    print("Central Composite Design alpha: ", np.round(alpha, 3))

elif design_parameters_dict["Design_Type"] == "Full_Factorial":

    design = pd.DataFrame(
        data = generate_full_factorial(
        variables_dict = variables_dict
        ),
        columns = variable_names
        )

    design["DataPointType"] = "FullFactorial"

elif design_parameters_dict["Design_Type"] == "2_Level_Full_Factorial":

    design = pd.DataFrame(
        data = generate_full_factorial(
        variables_dict = variables_dict
        ),
        columns = variable_names
        )

    design["DataPointType"] = "2_Level_Full_Factorial"

elif design_parameters_dict["Design_Type"] == "Plackett_Burman":

    design = GeneratePlackettBurman(
        num_dimensions = len(variables_dict),
        variable_names= variable_names,
        num_replicates = num_replicates
        )
    design["DataPointType"] = design_parameters_dict["Design_Type"]

else:
    print("Unknown Design Type")
    sys.exit()




# add sampling
if "Generate_Validation_Points" in design_parameters_dict:
    if design_parameters_dict["Generate_Validation_Points"] == True:

        train_test_split = 0.25

        num_samples = math.ceil((design.shape[0]/num_replicates) * train_test_split)

        LHC_Samples = pd.DataFrame(
            data = generate_latin_hypercube_samples(
            num_dimensions = len(variables_dict),
            num_samples = num_samples,
            num_replicates = num_replicates
            ),
            columns = variables_dict.keys()
            )
        LHC_Samples["DataPointType"] = "Validation"
        design = pd.concat([design, LHC_Samples])
        



design.reset_index(inplace=True, drop=True)


## assign condition id

# Initialise empty Condition_id Column
design["Condition_id"] = pd.Series(dtype="int64")

print(design)
# strip out the excess to create no replicates
no_duplicates = design.drop_duplicates().reset_index(drop=True)


# iterate over and assign if theres a match before incrementing
counter = int(1)
for i, no_des_row in no_duplicates.iterrows():
    for design_i, design_row in design.iterrows():
        #print(no_des_row[variable_names])
        #print(design_row[variable_names])
        if design_row[variable_names].equals(no_des_row[variable_names]):
            design.loc[design_i,"Condition_id"] = counter
    counter += 1
# make int.
design["Condition_id"] = design["Condition_id"].astype("int64")


if design_parameters_dict["Randomise"]:
    design = design.sample(frac=1, random_state=1233)


design.to_csv(project_path + "/Experiment_Designs/design.csv", index = None)

