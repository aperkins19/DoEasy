"""
This script cleans and tidys the rawdata set and returns a csv ready for plotting.
It determines what type of plate reader etc to correctly process the data
"""
### import

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import sys
#from sub_scripts.plotting_functions import *


# import experimental design parameters
#with open("./settings/design_parameters.json") as json_file:
#    design_parameters = json.load(json_file)

#import the project path
project_path = sys.argv[1]

# Load the CSV file
file_path = project_path + "Datasets/tidy_dataset.csv"  # Replace with your CSV file path
tidy_data = pd.read_csv(file_path)

# import variable component names
design_parameters_path = project_path + "/design_parameters.json"
design_parameters_dict = json.load(open(design_parameters_path, 'r'))
variables_dict = design_parameters_dict["Variables"]
variable_component_names = list(design_parameters_dict.keys())



## Initial trimming.
# drop negative control
plotting_data = tidy_data[tidy_data["Time"] <= 100]

plotting_data = plotting_data[
    [
        'Time',
        'Well',
        'RFUs',
        'Condition_id'
        ]
    ]

## time course

fig = plt.figure(figsize=(6, 5))
fig.suptitle(design_parameters_dict["Experiment_Name"] + " : Fluorescence of All Conditions Over Time", fontsize=15)


ax = sns.lineplot(
    data=plotting_data,
    y="RFUs",
    x="Time",
    hue="Condition_id"
)

# Set maximum number of ticks on each axis to 4
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

# Increase tick label size
ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust the size as needed

# Set new axis labels
ax.set_xlabel("Time (Mins)", fontsize=15)  # New x-axis label with adjusted font size
ax.set_ylabel("Relative Fluorescence Units (RFUs)", fontsize=15)  # New y-axis label with adjusted font size

# Legend and layout adjustments
legend = plt.legend(title="Runs", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(legend.get_title(), fontsize=15)  # Set the fontsize of the legend title
plt.tight_layout()

# Save the figure
plt.savefig(project_path + "Datasets/All_Conditions_over_Time.png")
plt.clf()
