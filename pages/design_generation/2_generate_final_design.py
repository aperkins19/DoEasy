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

coded_design = pd.read_csv(project_path + "/Experiment_Designs/design.csv")

real_values_df = pd.DataFrame()

## determine if screening design
if (design_parameters_dict["Design_Type"] == "Plackett_Burman") or (design_parameters_dict["Design_Type"] == "2_Level_Full_Factorial"):

    # save and remove the Condition_id & DataPointType column
    Condition_id = coded_design["Condition_id"]; coded_design.drop('Condition_id', axis=1, inplace=True)
    DataPointType = coded_design["DataPointType"]; coded_design.drop('DataPointType', axis=1, inplace=True)
    
    # Custom function for conditional replacement
    def conditional_replace(element):
        if element == '+':
            return replacement_dict['Max']
        elif element == '-':
            return replacement_dict['Min']
        else:
            return element

    for col_name, col_data in coded_design.items():

        # get replacement dict
        replacement_dict = variables_dict[col_name]
        # apply changes
        new_series = col_data.apply(conditional_replace)
        real_values_df[col_name] = new_series

    # reappend condition_id & DataPointType
    real_values_df["Condition_id"] = Condition_id
    real_values_df["DataPointType"] = DataPointType



elif (design_parameters_dict["Design_Type"] == "CCD") or (design_parameters_dict["Design_Type"] == "Full_Factorial") :
    for variable_name, variable_dict in variables_dict.items():

        original_series = coded_design[variable_name].copy()

        # Desired range for the dimension
        desired_range = (variable_dict["Min"], variable_dict["Max"])

        # Calculate scaling factor and midpoint
        scaling_factor = (desired_range[1] - desired_range[0]) / 2
        midpoint = sum(desired_range) / 2
        # Reverse the scaling for the dimension
        reversed_series = (original_series * scaling_factor) + midpoint

        real_values_df = pd.concat([real_values_df, reversed_series], axis = 1)

    # round all values to 2 decimal places
    real_values_df = real_values_df.round(2)

    real_values_df["DataPointType"] = coded_design["DataPointType"]
    real_values_df["Condition_id"] = coded_design["Condition_id"]





# round Discrete Variables.
discrete_variables = [key for key, value in variables_dict.items() if value.get("Type") == "Discrete"]


def round_discrete_columns(df, discrete_variables):
    """
    Rounds the columns in the DataFrame if their names are in the discrete_variables list.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    discrete_variables (list): A list of column names to be rounded.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns rounded.
    """
    for column_name in discrete_variables:
        if column_name in df.columns:

            print(column_name)
            print(df[column_name].apply(round))
            #df[column_name] = df[column_name].apply(round)
            # Rounding using Python's built-in round function
    return df

real_values_df = round_discrete_columns(real_values_df, discrete_variables)



real_values_df.to_csv(project_path + "/Experiment_Designs/design_real_values.csv", index=None)

#############################################################################################################################################
