import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import numpy as np

import sys


#import the project path 
project_path = sys.argv[1]


# import design
design = pd.read_csv(project_path + "/Datasets/tidy_dataset.csv")


# reorder axial
condition_id_order = [1,2,3,4,5,6,7,8,9, 15, 14, 12,10, 11,13]
design['Condition_id'] = pd.Categorical(design['Condition_id'], categories=condition_id_order, ordered=True)
design = design.sort_values(by='Condition_id')

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
        # Normalize numeric
        return (x - design_parameters_dict["Variables"][x.name]["Min"]) / (design_parameters_dict["Variables"][x.name]["Max"] - design_parameters_dict["Variables"][x.name]["Min"])

df_normalized = design_vars.apply(process_column)

# Create a figure
plt.figure(figsize=(8, 6))  # You can adjust the size as needed

# Add a title
if design_parameters_dict["Design_Type"] == "CCD":
    # Create a heatmap
    ax = sns.heatmap(df_normalized, annot=None, cmap='coolwarm', fmt='g')
    plt.title('Circumscribed Central Composite Design', fontsize=16)
    plt.ylabel('Experimental Conditions', fontsize=12)
    plt.xlabel('Design Variables', fontsize=12)

    # Define custom ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0 - ((design_parameters_dict["Specified_Alpha"] - 1)/2), 0, 0.5, 1, 1 + ((design_parameters_dict["Specified_Alpha"] - 1))/2])
    colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
    colorbar.set_label('Variable Datapoint', fontsize=12)

    plt.ylabel('Experimental Conditions', fontsize=12)

    #print(design_parameters_dict["Design_Type"])


elif design_parameters_dict["Design_Type"] == "Full_Factorial":
    ax = sns.heatmap(df_normalized, annot=design_vars, cmap='coolwarm', fmt='g')
    plt.title('Full Factorial Design', fontsize=16)
    plt.ylabel('Experimental Conditions', fontsize=12)
    plt.xlabel('Design Variables', fontsize=12)

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
plt.ylabel('Experimental Conditions', fontsize=12)






# Save the figure
plt.savefig(project_path + "/Experiment_Designs/design_heatmap.png", dpi=300, bbox_inches='tight')  # Adjust file name, DPI, and bbox_inches as needed
plt.savefig(project_path + "/Experiment_Designs/design_heatmap.svg", format="svg")  # Adjust file name, DPI, and bbox_inches as needed

# Display the plot
plt.show()

plt.clf()







### figure heatmap

# reshape data
df_normalized_T = df_normalized.reset_index(drop=True)
df_normalized_T = df_normalized_T[variables_list]
df_normalized_T = df_normalized_T.T



# Set up the figure
fig, ax = plt.subplots(figsize=(12, 2))

# Plot the heatmap
sns.heatmap(df_normalized_T, annot=None, cmap='coolwarm', fmt='g', ax=ax)

# redo the colour bar
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
colorbar.set_label('Level', fontsize=12)

# cosmetics

# Xtick labels at the center of each box
conditions = list(design["Condition_id"].unique())
tick_pos = [0.5 + i for i in range(df_normalized_T.shape[1])]  # Calculate positions
ax.set_xticks(tick_pos)  # Set tick positions
ax.set_xticklabels(conditions, ha='center')  # Set custom labels and align center

ax.set_xlabel("Condition")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


plt.tight_layout()
plt.savefig(project_path + "/Experiment_Designs/heatmap_for_fig.png", dpi=300, bbox_inches='tight')  # Adjust file name, DPI, and bbox_inches as needed
plt.savefig(project_path + "/Experiment_Designs/heatmap_for_fig.svg", format="svg")  # Adjust file name, DPI, and bbox_inches as needed


