import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json
import numpy as np

import sys


#import the project path 
project_path = sys.argv[1]
feature_to_model = sys.argv[2]

# import design
data = pd.read_csv(project_path + "/Datasets/tidy_dataset.csv")

# import the design_parameters_dict
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))
variables_dict = design_parameters_dict["Variables"]
variables_list = list(variables_dict.keys())

print(data)
######
centerpoints = data[data["DataPointType"] == "Center"].copy()


print(centerpoints)
print("Centerpoints Mean", centerpoints[feature_to_model].mean())
print("Centerpoints STD" ,centerpoints[feature_to_model].std())


axial = data[data["DataPointType"] == "Axial"].copy()
axial = data[data["Condition_id"] == 15].copy()




top = data[data["Condition_id"] == 9].copy()
print("best", top[feature_to_model].mean())

worst = data[data["Condition_id"] ==12].copy()
print("worst", worst[feature_to_model].mean())


#
# design_vars = (design[variables_list]).drop_duplicates().reset_index(drop=True)
#
#
#
#
# def process_column(x):
#     if x.dtype == 'object':  # checks if the column is of string type
#         # If string, double check what type of experimental type
#         if (design_parameters_dict["Design_Type"] == "Plackett_Burman") or design_(parameters_dict["Design_Type"] == "2_Level_Full_Factorial"):
#             # Label encoding
#             encoded, unique = pd.factorize(x)
#         # returns encoded series
#         return pd.Series(encoded, dtype='float64')
#     else:
#         # Normalize numeric columns
#         return (x - np.min(x)) / (np.max(x) - np.min(x))
#
# df_normalized = design_vars.apply(process_column)
#
# # Create a figure
# plt.figure(figsize=(8, 6))  # You can adjust the size as needed
#
# # Add a title
# if design_parameters_dict["Design_Type"] == "CCD":
#     # Create a heatmap
#     ax = sns.heatmap(df_normalized, annot=None, cmap='coolwarm', fmt='g')
#     plt.title('Circumscribed Central Composite Design', fontsize=16)
#     plt.xlabel('Design Variables (Values in mM)', fontsize=12)
#
#     # Define custom ticks
#     colorbar = ax.collections[0].colorbar
#     colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
#     colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
#     colorbar.set_label('Variable Datapoint', fontsize=12)
#
#     plt.ylabel("test", fontsize=12)
#     #print(design_parameters_dict["Design_Type"])
#
#
# elif design_parameters_dict["Design_Type"] == "Full_Factorial":
#     ax = sns.heatmap(df_normalized, annot=design_vars, cmap='coolwarm', fmt='g')
#     plt.title('Full Factorial Design', fontsize=16)
#     plt.xlabel('Design Variables (Values in mM)', fontsize=12)
#
#     #### this needs to be revamped!!!!!!
#     # Define custom ticks
#     colorbar = ax.collections[0].colorbar
#     colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
#     colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
#     colorbar.set_label('Variable Datapoint', fontsize=12)
#
# elif (design_parameters_dict["Design_Type"] == "Plackett_Burman") or (design_parameters_dict["Design_Type"] == "2_Level_Full_Factorial"):
#
#     ax = sns.heatmap(df_normalized, cmap='coolwarm')
#     plt.title('Plackett-Burman Design', fontsize=16)
#     plt.xlabel('Design Variables', fontsize=12)
#
#     # Define custom ticks
#     colorbar = ax.collections[0].colorbar
#     colorbar.set_ticks([0, 1])
#     colorbar.set_ticklabels(['Low', 'High'])
#     colorbar.set_label('Variable Datapoint', fontsize=12)
#
# # Add axis labels
# plt.ylabel('Experimental Conditions', fontsize=12)
#
#
#
#
#
#
# # Save the figure
# plt.savefig(project_path + "/Experiment_Designs/design_heatmap.png", dpi=300, bbox_inches='tight')  # Adjust file name, DPI, and bbox_inches as needed
# plt.savefig(project_path + "/Experiment_Designs/design_heatmap.svg", format="svg")  # Adjust file name, DPI, and bbox_inches as needed
#
# # Display the plot
# plt.show()
#
# plt.clf()
#
#
#
#
#
#
#
# ### figure heatmap
#
# # reshape data
# df_normalized_T = df_normalized.reset_index(drop=True)
# df_normalized_T = df_normalized_T[variables_list]
# df_normalized_T = df_normalized_T.T
#
#
#
# # Set up the figure
# fig, ax = plt.subplots(figsize=(12, 2))
#
# # Plot the heatmap
# sns.heatmap(df_normalized_T, annot=None, cmap='coolwarm', fmt='g', ax=ax)
#
# # redo the colour bar
# colorbar = ax.collections[0].colorbar
# colorbar.set_ticks([0, 0.15, 0.5, 0.85, 1])
# colorbar.set_ticklabels(['Low Axial', 'Low', 'Center', 'High', 'High Axial'])
# colorbar.set_label('Level', fontsize=12)
#
# # cosmetics
#
# # Xtick labels at the center of each box
# conditions = list(design["Condition_id"].unique())
# tick_pos = [0.5 + i for i in range(df_normalized_T.shape[1])]  # Calculate positions
# ax.set_xticks(tick_pos)  # Set tick positions
# ax.set_xticklabels(conditions, ha='center')  # Set custom labels and align center
#
# ax.set_xlabel("Condition")
# ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#
#
# plt.tight_layout()
# plt.savefig(project_path + "/Experiment_Designs/heatmap_for_fig.png", dpi=300, bbox_inches='tight')  # Adjust file name, DPI, and bbox_inches as needed
# plt.savefig(project_path + "/Experiment_Designs/heatmap_for_fig.svg", format="svg")  # Adjust file name, DPI, and bbox_inches as needed
#

