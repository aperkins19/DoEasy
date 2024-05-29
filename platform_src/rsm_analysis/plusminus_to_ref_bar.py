import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys
import json 
from matplotlib.ticker import MaxNLocator

#import the project path 
project_path = sys.argv[1]
savepath = sys.argv[2]
feature_to_model = sys.argv[3]

# Load the CSV file
file_path = project_path + "Datasets/tidy_dataset.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# import the design_parameters_dict
design_parameters_path = project_path + "/design_parameters.json"
design_parameters_dict = json.load(open(design_parameters_path, 'r'))
input_variables_dict = design_parameters_dict["Variables"]
response_variables_dict = design_parameters_dict["Response_Variables"]

# get variables
X_variables = list(input_variables_dict.keys())

##########################################
# get reference_condition_id_int

# check if reference condition is designated or not
if design_parameters_dict["Reference_Condition"] == "Not Designated":
    print("")
    print("A reference condition has not been designated in the experiment configuration, checking if design contains centerpoints..")
    
    # if does contain true (all false is 0)
    if data["DataPointType"].str.contains("Center").sum() > 0:
        reference_condition_id_int = data[data["DataPointType"] == "Center"]["Condition_id"].unique()[0]
        print("Design contains Centerpoints. Designating as reference condition. Condition ID: " + str(reference_condition_id_int))
    else:
        user_input = input("Please enter the Condition # to use as the reference condition (e.g. '9'): ")
        reference_condition_id_int = int(user_input)
        print("Condition ID: " + str(reference_condition_id_int))
else:
    pass
    ### need to add functionality to if design_parameters_dict["Reference_Condition"]: { asdfasdf }


########################################
# data calculations
# get meanY_of_relative_compositon
relevant_columns = ["Condition_id", feature_to_model] + X_variables
relative_data_df = data.copy()
relative_data_df = relative_data_df[relevant_columns].drop_duplicates().reset_index(drop=True)
meanY_of_relative_compositon = relative_data_df[relative_data_df["Condition_id"] == reference_condition_id_int][feature_to_model].mean()

# create relative_data_df
relative_data_df["%_Change_over_Reference_Composition"] = ((relative_data_df[feature_to_model] / meanY_of_relative_compositon) - 1) * 100
print(relative_data_df)

#################################
## vertical barplot
fig = plt.figure(figsize=(12, 7))
ax = sns.barplot(
        data = relative_data_df,
        x="%_Change_over_Reference_Composition",
        y="Condition_id",
        orient="h",
        saturation=1,
        #color=color,
        width=0.75,
    )
# Customizing the plot
plt.xlabel('% Change')
plt.ylabel('Condition ID')
plt.title("% Change of each condition relative to reference")

plt.tight_layout()
fig.savefig(savepath + "/percent_change_over_ref_condition_barplot.png")
