### import

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
import json


from sub_scripts.response_surface_methadology_modelling import LinearModel, PolyModel



#import the project path 
project_path = sys.argv[1]
savepath = sys.argv[2]
feature_to_model = sys.argv[3]


# Load the CSV file
file_path = project_path + "Datasets/tidy_dataset.csv"  # Replace with your CSV file path
tidy_data = pd.read_csv(file_path)

# import experimental design
design_path = project_path + "Experiment_Designs/design_real_values.csv"
exp_design = pd.read_csv(design_path)

# import variable component names
design_parameters_path = project_path + "/design_parameters.json"
design_parameters_dict = json.load(open(design_parameters_path, 'r'))
variables_dict = design_parameters_dict["Variables"]
variable_component_names = list(variables_dict.keys())

# import variable component names
X_names = variable_component_names
Y_names = feature_to_model

# Experiment names
experiment_description =  design_parameters_dict["Experiment_Name"] +"_"+Y_names[0]


# dynamically get 
plotting_data = tidy_data[X_names + [Y_names] + ["DataPointType"] + ["Condition_id"]].copy().reset_index(drop=True)
# colour code based on if Validation
plotting_data["IsValidation"] = plotting_data["DataPointType"] == "Validation"

figsize = (12, 5)
plt.figure(figsize=figsize)

# Ensure dodge=True for violinplot for proper alignment when using hue
sns.violinplot(
    x="Condition_id",
    y=Y_names,
    #hue="IsValidation",
    data=plotting_data,
    alpha=0.5,
    dodge=True  # This ensures separate violins for each hue category
)


# add a buffer around the min and max for visual clarity
buffer = (plotting_data[Y_names].max() - plotting_data[Y_names].min()) * 0.25
plt.ylim(plotting_data[Y_names].min() - buffer, plotting_data[Y_names].max() + buffer) 

# Now plot the stripplot; you may adjust jitter as needed
sns.stripplot(
    x="Condition_id",
    y=Y_names,
    data=plotting_data,
    jitter=True,  # Jitter helps in spreading out the points
    alpha=1,
    color="b",
    dodge=True  # Though stripplot doesn't have dodge, setting this in anticipation of future compatibility or custom handling
)


# Add a horizontal line at y = value
plt.axhline(y=0, color='r', alpha=0.2, linestyle='--')



plt.title("Intra-Condition Variance")
plt.tight_layout()

# Save the plot
plt.savefig(f"{savepath}/intra_condition_variance_kde_datapoints.png")
plt.savefig(f"{savepath}/intra_condition_variance_kde_datapoints.svg", format="svg")
plt.clf()


###############################
# noise of all conditions
fontsize = 20
# Define your desired figsize (width, height in inches)
figsize = (10, 6)
plt.figure(figsize=figsize)
fig, ax = plt.subplots(figsize=figsize)

sns.barplot(
    data=tidy_data,
    x="Condition_id",
    y=feature_to_model,
    color="black",
    alpha=0.75,
    errcolor="red",
    ax=ax
    )

if design_parameters_dict["Technical_Replicates"] > 1:

    sns.stripplot(
        data= tidy_data,
        x="Condition_id",
        y=feature_to_model,
        jitter=True,  # Jitter helps in spreading out the points
        alpha=1,
        color="g",
        size=10,
        dodge=True,
        ax=ax
        )

ax.set_yticks(np.round(np.linspace(0,tidy_data[feature_to_model].max()*1.2,4),2))

ax.set_xlabel('Condition ID', fontsize=fontsize)
# Keep the existing x-tick labels but change their fontsize
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)


plt.ylabel(feature_to_model +" ("+ design_parameters_dict["Response_Variables"][feature_to_model]["Units"] + ")", fontsize = fontsize)

plt.title(feature_to_model + " of Each Condition (Mean & 95% CI)", fontsize = fontsize)

plt.tight_layout()

plt.savefig(savepath + "/" + "all_conditions_bar.png")
plt.savefig(f"{savepath}/all_conditions_bar.svg", format="svg")
plt.clf()
