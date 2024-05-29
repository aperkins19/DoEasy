### import

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
import json
import math

from sub_scripts.response_surface_methadology_modelling import LinearModel, PolyModel



#import the project path 
project_path = sys.argv[1]
savepath = sys.argv[2]
feature_to_model = sys.argv[3]

print()
print("feature to model")
print(feature_to_model)
print()

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

print(tidy_data)
# Calculate the SEM for each Condition_id
sem_values = tidy_data.groupby('Condition_id')[feature_to_model].sem().reset_index(name='SEM')

# Calculate the Mean for each Condition_id
mean_values = tidy_data.groupby('Condition_id')[feature_to_model].mean().reset_index(name='Mean')


# Merge the SEM values back into the original DataFrame
tidy_data = tidy_data.merge(sem_values, on='Condition_id')

# Merge the Mean values back into the original DataFrame
tidy_data = tidy_data.merge(mean_values, on='Condition_id')


# Define your desired figsize (width, height in inches)
figsize = (12, 5)
plt.figure(figsize=figsize)
fig, ax = plt.subplots(figsize=figsize)

sns.barplot(
    data=tidy_data,
    x="Condition_id",
    y=feature_to_model,
    color="black",
    alpha=0.5,
    errorbar=None,
    ax=ax
    )


#### add error bars
# Get the current axis
ax = plt.gca()

# Calculate the positions of the bars to know where to place the error bars
bar_centers = [p.get_x() + p.get_width() / 2 for p in ax.patches]

# Calculate the mean
# Manually add the error bars using the pre-calculated SEM values
ax.errorbar(
    x=bar_centers,
    y=list(tidy_data["Mean"].unique()),
    yerr=list(tidy_data["SEM"].unique()),
    fmt='none',
    capsize=5,
    capthick = 3,
    color='black',
    elinewidth = 3
    )

##### add real data points

sns.stripplot(
    data=tidy_data,
    x="Condition_id",
    y=feature_to_model,
    jitter=0.3,  # Jitter helps in spreading out the points
    alpha=0.7,
    size=12,
    dodge=True,
    marker="o",
    color = "red",
    ax=ax
)

## add base line

# calculate sem of GFP baseline
baseline = pd.Series([46.39, 45.77, 47.32])

# Define the top and bottom boundaries for the shaded area on the y-axis
y_bottom = baseline.mean() - baseline.sem()
y_top = baseline.mean() + baseline.sem()

print()
print(baseline.mean() + baseline.sem())
print(baseline.mean())
print(baseline.mean() - baseline.sem())

# Add a horizontal line at y = value
ax.axhline(y=baseline.mean(), color='black', alpha=1, linestyle='--').set_dashes([10, 5])


baseline_x = np.arange(
    (tidy_data["Condition_id"].min() -2),
    (tidy_data["Condition_id"].max()+1),
    1
    )

# # Add a horizontal shaded area across the plot
# ax.fill_between(
#     x = baseline_x,
#     y1 = y_bottom,
#     y2 = y_top,
#     color = 'black',
#     alpha = 0.3
#     )

# Remove x-ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlabel('')

y_ceil = math.ceil(baseline.mean() * 2)

ax.set_yticks([0, math.ceil(baseline.mean()), 90])
ax.tick_params(axis='y', labelsize=20)

plt.ylabel((feature_to_model + " (" + design_parameters_dict["Response_Variables"][feature_to_model]["Units"] +")"), fontsize = 20)

ax.set_xlim(
    (tidy_data["Condition_id"].min()-1.5),
    (tidy_data["Condition_id"].max()-0.5)
    )

#plt.title("Protein Yield of Each Condition (Mean & 95% CI)")

plt.tight_layout()

plt.savefig(savepath + "/" + "all_conditions_bar.png")
plt.savefig(f"{savepath}/all_conditions_bar.svg", format="svg")
plt.clf()
