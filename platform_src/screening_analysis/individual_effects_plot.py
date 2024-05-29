import sys
import json
import pickle

import pandas as pd
import numpy as np 
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Function to calculate the effect of each factor
def calculate_effects(data, factor_names, response_variable):
    effects = {}
    for factor in factor_names:

        # skip if response variable
        if factor == response_variable:
            pass 
        else:
            # Calculate the mean response for high and low levels of the factor
            mean_high = data[data[factor] == "+"][response_variable].mean()
            mean_low = data[data[factor] == "-"][response_variable].mean()

            # The effect is half the difference between these means
            effect = (mean_high - mean_low) / 2
            effects[factor] = effect

    return effects


def generate_effects_df(coded_design_with_response_values: pd.DataFrame, input_variables: list, response_variables: list):

    # generate effects for each response variable
    effects_df = pd.DataFrame()
    for i, response_variable in enumerate(response_variables):

        # generate factor names
        factor_names = input_variables.copy()
        factor_names.append(response_variable)
        
        # generate response variable specific df
        response_variable_specific_df = coded_design_with_response_values.copy()
        response_variable_specific_df = response_variable_specific_df[factor_names]

        # calculate effects
        effects_dict = calculate_effects(response_variable_specific_df, factor_names, response_variable)
        
        # add dict to df
        effects_df[response_variable] = pd.Series(effects_dict)
    
    return effects_df


#import the project path 
project_path = sys.argv[1]

# import design parameters
design_parameters_dict = json.load(open(project_path + "design_parameters.json", "r"))

# import dataset
tidy_data = pd.read_csv(project_path+"/Datasets/tidy_dataset.csv")

# import coded design
coded_design = pd.read_csv(project_path+"/Experiment_Designs/design.csv")

# get variable name lists
input_variables = list(design_parameters_dict["Variables"].keys())
response_variables = list(design_parameters_dict["Response_Variables"].keys())

# generate coded design with response values
# Merging coded_design with response variables from tidy_data via condition_id
coded_design_with_response_values = coded_design.merge(tidy_data[["Condition_id"]+response_variables], on="Condition_id", how="right")

individual_effects_df = pd.DataFrame(columns=["Name", "Min", "Max", "Response_Variable"])

# iterate over response and input variables
for response_variable in response_variables:
    for i, input_variable in enumerate(input_variables):

        # grab the relevant input and output variables and merge
        input_series = coded_design_with_response_values[input_variable]
        response_series = coded_design_with_response_values[response_variables]
        individual = pd.concat([input_series, response_series], axis = 1)

        # calculate the max and min by averaging
        individual_effect_series = pd.Series([
                input_variable,
                individual[individual[input_variable] == "+"][response_variable].mean(),
                individual[individual[input_variable] == "-"][response_variable].mean()
            ], index = ["Name", "Min", "Max"]).to_frame().transpose()

        # assign the response variable and append to the main df
        individual_effect_series["Response_Variable"] = response_variable
        individual_effects_df = pd.concat([individual_effects_df, individual_effect_series], axis=0)
        

individual_effects_df.reset_index(drop=True, inplace=True)

# melt to get tidy data
individual_effects_df = individual_effects_df.melt(
    id_vars = ["Name", "Response_Variable"], #  col to keep the same
    var_name = "Range", # new col for the old column names
    value_name = "Value", # new col name for values
    )



# # generate effects for each response variable
# effects_df = generate_effects_df(
#     coded_design_with_response_values = coded_design_with_response_values,
#     input_variables = input_variables,
#     response_variables = response_variables
# )

# plot all effects

# Define a color palette
palette = sns.color_palette("colorblind", n_colors=len(individual_effects_df['Response_Variable'].unique()))


# Number of plots
num_subplots = len(input_variables)

# Determine the layout of the subplots
rows = int(np.ceil(num_subplots / 2))  # Adjust the number of rows as needed
cols = 2

# Create the subplots
fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))

# # Flatten the axes array for easy indexing
axes = axes.flatten()

# # Creating individual subplots
for ax, input_variable in zip(axes, input_variables):
    input_var_df = individual_effects_df[individual_effects_df["Name"] == input_variable].copy()
    # add column for min:0, max:0
    input_var_df['Min_Max_Flag'] = input_var_df['Range'].apply(lambda x: 0 if x == 'Min' else 1)

    sns.scatterplot(
        data = input_var_df,
        x = "Min_Max_Flag",
        y = "Value",
        hue = "Response_Variable",
        palette=palette,
        ax = ax
    )
    sns.lineplot(
        data = input_var_df,
        x = "Min_Max_Flag",
        y = "Value",
        hue = "Response_Variable",
        palette=palette,
        ax = ax
    )

    ax.set_title(input_variable)
    ax.set_xlabel("Range")
    ax.set_ylabel("Value")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Min","Max"])
    ax.legend().remove()  # Remove individual legends


# Create a shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(input_variables))

# Adjust layout to make room for the shared legend
fig.suptitle("Individual Effects Plots", fontsize = 20)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend

fig.savefig(project_path + "/Output/Individual_Effects_Plots.png")






#   print(row)
#     # effects plot
#     sns.barplot(
#         data=plotting_df,
#         x = "Factor",
#         y = "Effect",
#         ax=ax)

#     ax.set_title(col_name)
#     ax.set_xlabel("Input Variables")
#     ax.tick_params(axis='x', rotation=90)
#     ax.set_xlabel("Factors")
#     ax.grid(axis="y")


# # Hide any unused subplots
# for j in range(num_subplots, len(axes)):
#     fig.delaxes(axes[j])

# fig.suptitle("Main Effects of the Input Variables to Each Response Variable")
# fig.tight_layout()
# fig.savefig(project_path + "/Output/Main_Effects_Plots.png")
