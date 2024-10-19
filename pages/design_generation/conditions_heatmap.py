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
        if design_parameters_dict["Design_Type"] == "2_Level_Full_Factorial":
            # Label encoding
            encoded, unique = pd.factorize(x)
            # returns encoded series
            return pd.Series(encoded, dtype='float64')

        # If string, double check what type of experimental type
        elif (design_parameters_dict["Design_Type"] == "Plackett_Burman"):

            mapping = {
                        design_parameters_dict["Variables"][x.name]["Max"]: 1,
                        design_parameters_dict["Variables"][x.name]["Min"]: 0
                        }
            # Label encoding
            x_mapped = x.replace(mapping)
            # returns encoded series
            return pd.Series(x_mapped, dtype='float64')

    else:
        # Normalize numeric columns
        return (x - np.min(x)) / (np.max(x) - np.min(x))

df_normalized = design_vars.apply(process_column)



# Create a figure
plt.figure(figsize=(8, 9))  # You can adjust the size as needed
fontsize = 20

# Add a title
if design_parameters_dict["Design_Type"] == "CCD":
    # Create a heatmap
    ax = sns.heatmap(df_normalized, annot=None, cmap='coolwarm', fmt='g')
    plt.title('Circumscribed Central Composite Design', fontsize=fontsize)
    plt.xlabel('Input Variables', fontsize=fontsize)
    plt.ylabel('Experimental Conditions', fontsize=fontsize)

    # Customize the y-ticks
    tick_positions = [x + 0.5 for x in range(len(design["Condition_id"].drop_duplicates()))]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(design["Condition_id"].drop_duplicates(), fontsize = fontsize-8)


    # Customize the x-ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize-4)  # Rotate x-ticks and set font size


    # Define custom ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
    colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'], fontsize=fontsize-7)
    colorbar.set_label('Coded Datapoint', fontsize=fontsize)

elif design_parameters_dict["Design_Type"] == "Full_Factorial":
    ax = sns.heatmap(df_normalized, annot=design_vars, cmap='coolwarm', fmt='g')
    plt.title('Full Factorial Design', fontsize=16)
    plt.xlabel('Input Variables', fontsize=12)
    plt.ylabel('Experimental Conditions', fontsize=12)

    # Customize the y-ticks
    tick_positions = [x + 0.5 for x in range(len(design["Condition_id"].drop_duplicates()))]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(design["Condition_id"].drop_duplicates())

    #### this needs to be revamped!!!!!!
    # Define custom ticks
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
    colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
    colorbar.set_label('Variable Datapoint', fontsize=12)

elif (design_parameters_dict["Design_Type"] == "Plackett_Burman") or (design_parameters_dict["Design_Type"] == "2_Level_Full_Factorial"):


    # Define a custom colormap for binary data
    cmap = sns.color_palette(["#3498db", "#e74c3c"], as_cmap=True)


    # Create the heatmap
    ax = sns.heatmap(df_normalized, cmap=cmap, cbar=False)

    # Setting the title and axis labels
    plt.title('Plackett-Burman Design', fontsize=16)
    plt.ylabel('Experimental Conditions', fontsize=12)
    plt.xlabel('Design Variables', fontsize=12)

    # Changing y-tick labels to a range from 1 to the number of rows
    num_rows = df_normalized.shape[0]
    ax.set_yticklabels(range(1, num_rows + 1))

    # Create a custom legend
    import matplotlib.patches as mpatches
    blue_patch = mpatches.Patch(color='#3498db', label='-')
    red_patch = mpatches.Patch(color='#e74c3c', label='+')
    plt.legend(handles=[blue_patch, red_patch], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')


    #
    # ax = sns.heatmap(df_normalized, cmap='coolwarm')
    # num_rows = df_normalized.shape[0]
    # ax.set_yticklabels(range(1, num_rows + 1))
    # plt.title('Plackett-Burman Design', fontsize=16)
    # plt.ylabel('Experimental Conditions', fontsize=12)
    # plt.xlabel('Design Variables', fontsize=12)
    #
    # # Define custom ticks
    # colorbar = ax.collections[0].colorbar
    # colorbar.set_ticks([0, 1])
    # colorbar.set_ticklabels(['-', '+'])
    # #colorbar.set_label('Variable Datapoint', fontsize=12)


# Save the figure
plt.savefig(project_path + "/Experiment_Designs/design_heatmap.png", dpi=300, bbox_inches='tight')  # Adjust file name, DPI, and bbox_inches as needed

# Display the plot
plt.show()
