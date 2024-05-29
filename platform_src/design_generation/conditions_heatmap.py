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



design_vars = (design[variables_list]).drop_duplicates().reset_index(drop=True)


def process_column(x):
    if x.dtype == 'object':  # checks if the column is of string type
        # If string, double check what type of experimental type
        if (design_parameters_dict["Design_Type"] == "Plackett_Burman") or design_(parameters_dict["Design_Type"] == "2_Level_Full_Factorial"):
            # Label encoding
            encoded, unique = pd.factorize(x)
        # returns encoded series
        return pd.Series(encoded, dtype='float64')
    else:
        # Normalize numeric columns
        return (x - np.min(x)) / (np.max(x) - np.min(x))

df_normalized = design_vars.apply(process_column)

# Create a figure
plt.figure(figsize=(8, 6))  # You can adjust the size as needed

# Add a title
if design_parameters_dict["Design_Type"] == "CCD":
    # Create a heatmap
    ax = sns.heatmap(df_normalized, annot=None, cmap='coolwarm', fmt='g')
    plt.title('Circumscribed Central Composite Design', fontsize=16)
    plt.xlabel('Design Variables (Values in mM)', fontsize=12)

    # Define custom ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
    colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
    colorbar.set_label('Variable Datapoint', fontsize=12)

elif design_parameters_dict["Design_Type"] == "Full_Factorial":
    ax = sns.heatmap(df_normalized, annot=design_vars, cmap='coolwarm', fmt='g')
    plt.title('Full Factorial Design', fontsize=16)
    plt.xlabel('Design Variables (Values in mM)', fontsize=12)

    #### this needs to be revamped!!!!!!
    # Define custom ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
    colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
    colorbar.set_label('Variable Datapoint', fontsize=12)

elif (design_parameters_dict["Design_Type"] == "Plackett_Burman") or (design_parameters_dict["Design_Type"] == "2_Level_Full_Factorial"):

    ax = sns.heatmap(df_normalized, cmap='coolwarm')
    plt.title('Plackett-Burman Design', fontsize=16)
    plt.xlabel('Design Variables', fontsize=12)

    # Define custom ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(['Low', 'High'])
    colorbar.set_label('Variable Datapoint', fontsize=12)

# Add axis labels
plt.ylabel('Experimental Runs', fontsize=12)






# Save the figure
plt.savefig(project_path + "/Experiment_Designs/design_heatmap.png", dpi=300, bbox_inches='tight')  # Adjust file name, DPI, and bbox_inches as needed

# Display the plot
plt.show()