import json
import pandas as pd
import math
import numpy as np
import sys
from auxillary_scripts.rsm_generation import *
from auxillary_scripts.calculators import *


#############################################################

#import the project path 
project_path = sys.argv[1]

experiment_designs_path = project_path + "/Experiment_Designs/"

#############################################################



# import the design_parameters_dict
design_parameters_path = project_path + "/design_parameters.json"
design_parameters_dict = json.load(open(design_parameters_path, 'r'))
variables_dict = design_parameters_dict["Variables"]


design = pd.read_csv(experiment_designs_path+ "design_real_values.csv")


# generate well lists
#### Assign Plates and Plate Wells

# generate the well well_list_384
well_list_384 = generate_well_list_384(spacing = design_parameters_dict["Well_Spacing"])
plate_capacity = len(well_list_384)

#  store the number of runs - the num of rows of the df
num_of_runs = design.shape[0]

# if there are fewer experiments than or exactly 384 or 77 or whatever (the plate capacity), then just label the with enough wells
if num_of_runs <= plate_capacity:
    design['Well'] = well_list_384[:num_of_runs]
    design['Plate'] = 1

    plates_required = 1
    # calculate runs per plate
    runs_per_plate = int(num_of_runs/plates_required)

elif num_of_runs > plate_capacity:

    # divides by 384 and rounds up to next integer
    plates_required = math.ceil(num_of_runs / plate_capacity)

    # calculate runs per plate
    runs_per_plate = int(num_of_runs/plates_required)

    # build lists of wells and plates to append to df as columns
    exp_wells = []
    exp_plates = []

    for plate in range(1, plates_required+1,1):

        for run in range(1, runs_per_plate+1,1):
            exp_plates.append(plate)
            exp_wells.append(well_list_384[run-1])

    design['Well'] = exp_wells
    design['Plate'] = exp_plates


else:
    raise Exception("There is an error with the number of experiments")



design.to_csv(experiment_designs_path + "design_real_values.csv", index=None)
