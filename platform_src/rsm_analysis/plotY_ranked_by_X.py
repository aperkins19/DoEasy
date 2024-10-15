import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys
import json 
from matplotlib.ticker import MaxNLocator
import itertools
import os
import shutil

def get_user_choice(options):
    """
    Display options to the user and get their choice.

    :param options: List of options to choose from.
    :return: Selected option.
    """
    # Display options to the user
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        try:
            # Get user input and convert to integer
            user_input = int(input("Enter your choice (number): "))

            # Validate input
            if 1 <= user_input <= len(options):
                return options[user_input - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")

        except ValueError:
            # Handle non-integer input
            print("Invalid input, please enter a number.")


def plot_ranked_permutations_with_subplots_fig(data_df, list_of_permutations, plot_id, save_dir):

    # initialise figure and subplots
    # Number of plots
    

    num_subplots = len(list_of_permutations)

    # Create the subplots
    if num_subplots == 8:
        rows = 4
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
        axes = axes.flatten()

    elif num_subplots == 4:
        rows=2
        cols=2
        fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 4))
        axes = axes.flatten()

    elif num_subplots == 3:
        rows=2
        cols=2
        fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
        axes = axes.flatten()


    # sort by permutation
    for ax, (i, permutation) in zip(axes, enumerate(list_of_permutations)):
        permutation_data = data_df.sort_values(by=list(permutation)).reset_index()
        permutation_data["index"] = permutation_data["index"].astype(str)

        #################################
        # subplot

        sns.barplot(
            data = permutation_data,
            x = "Condition_id",
            y = feature_to_model,
            hue = permutation[0],
            ax = ax,
            #errorbar=None,
            dodge=False
            )

        # Join the tuple elements into a string with ', ' as the separator
        permutation_string = ", ".join(permutation)
        ax.set_title("Ranked by: "+permutation_string)

        ax.legend(
            title=permutation[0]+" " + input_variables_dict[permutation[0]]["Units"],
            bbox_to_anchor=(1, 1)
            )
        
    # remove unused subplots
    for j in range(len(list_of_permutations), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(feature_to_model+" ranked by input permutations"+plot_id, fontsize = 20)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
    plt.savefig(save_dir + "/" + feature_to_model+"_ranked_by_input_permutations"+plot_id+".png")
    plt.close()  
    


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

########################################
# data calculations
# get meanY_of_relative_compositon
relevant_columns = ["Condition_id", feature_to_model] + X_variables
trimmed_data_df = data.copy()
trimmed_data_df = trimmed_data_df[relevant_columns].drop_duplicates().reset_index(drop=True)


# Generate permutations
# Generate all permutations
all_permutations = list(itertools.permutations(X_variables))


# check if too many permutations
if len(X_variables) >= 4:
    print("")
    print("Due to there being", len(X_variables), "input variables, there are", len(all_permutations), "possible permumations. This is too many to plot in one figure.")
    print("Would you like to: ")


    # get user choice
    options = ["[1] Rank with respect to one specific variable", "[2] Plot all them but over different figures.", "[3] Exit"]
    user_choice = get_user_choice(options)

    if user_choice == options[0]:
        print(f"You have selected: {user_choice}")


    elif user_choice == options[1]:
        print(f"You have selected: {user_choice}")

        # Check if the directory already exists
        plots_dir_name = savepath + "/Ranked_Plots"

        if not os.path.exists(plots_dir_name):
            # Create the directory
            os.mkdir(plots_dir_name)
            
        else:
            # if exists, delete and remake
            shutil.rmtree(plots_dir_name)
            os.mkdir(plots_dir_name)

        # determine subplots per figure
        print(trimmed_data_df["Condition_id"].max())

        if trimmed_data_df["Condition_id"].max() <= 16:
            subplots_per_fig = 8
        elif trimmed_data_df["Condition_id"].max() <= 28:
            subplots_per_fig = 6
        elif trimmed_data_df["Condition_id"].max() <= 48:
            subplots_per_fig = 3

        # split permutations across groups of 8
        sub_lists = [all_permutations[i:i + subplots_per_fig] for i in range(0, len(all_permutations), subplots_per_fig)]

        print()
        print("The", len(all_permutations),"will be split into a total of",len(sub_lists),"figures with",subplots_per_fig,"subplots each..")
        for i, list_of_permutations in enumerate(sub_lists):
            fig_num = i+1
            plot_ranked_permutations_with_subplots_fig(trimmed_data_df, list_of_permutations, "_Fig#_"+str(fig_num)+"_", plots_dir_name)
            print("Completed Plot ("+str(fig_num)+"/"+str(len(sub_lists))+")")


        


    elif user_choice ==  options[0]:
        print(f"You have selected: {user_choice}")
else:
    plot_ranked_permutations_with_subplots_fig(trimmed_data_df, all_permutations, "_All_Permutations_", savepath)
