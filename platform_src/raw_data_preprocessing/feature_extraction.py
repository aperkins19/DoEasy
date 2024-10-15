import pandas as pd
import numpy as np
import sympy as sy 
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#from sub_scripts.response_surface_methadology_modelling import *
from sub_scripts.feature_extraction import AsymmetricSigmoidFitter, GompertzFitter, NaturalLogFitter

import matplotlib.pyplot as plt

import sys

#import the project path 
project_path = sys.argv[1]

# Load the CSV file
file_path = project_path + "Datasets/tidy_dataset.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

print(data.head())


# import variable component names
design_parameters_path = project_path + "/design_parameters.json"
design_parameters_dict = json.load(open(design_parameters_path, 'r'))
variables_dict = design_parameters_dict["Variables"]
variable_component_names = list(design_parameters_dict.keys())

# get unique conditions
#unique_conditions = data[variable_component_names].drop_duplicates().reset_index()
#for i, condition in unique_conditions.iterrows():

#    one_condition = data[(data["MG_Glut"] == condition["MG_Glut"])&(data["K_Glut"]==condition["K_Glut"])&(data["DTT"]==condition["DTT"])]

#    one_condition = one_condition[one_condition["Time"]< 200]



# get unique wells
unique_wells = data["Well"].unique()



# initalise empty df
data_with_features = pd.DataFrame()

for well in unique_wells:

    # select just that well
    subset = data[data["Well"] == well]

    subset = subset[subset["Time"] > 0]
    subset = subset[subset["Time"]< 100]

    #raw_data_path = project_path + "Datasets/RawData/input_raw_data_dataset.csv"
    x = subset["Time"]
    y = subset["GFP_uM"]



    # Initialize variables to store the best fitter and the lowest MSE
    best_fitter = None
    lowest_mse = float('inf')



    try:
        # initialise fitter
        asym_fitter = AsymmetricSigmoidFitter()
        # Try fitting with Asymmetric Sigmoid Function
        asym_fitter.fit(x, y)
        asym_fitter.get_fit_metrics(x, y)

        if asym_fitter.mse < lowest_mse:
            best_fitter = asym_fitter
            lowest_mse = asym_fitter.mse

    except Exception as e:
        pass
        #print(f"Gompertz Fitting failed due to {e}. Condition: {subset['Condition'].unique()}")

    try:
        # Fit Gompertz function
        gompertz_fitter = GompertzFitter()
        gompertz_fitter.fit(x, y)
        gompertz_fitter.get_fit_metrics(x, y)


        if gompertz_fitter.mse < lowest_mse:
            best_fitter = gompertz_fitter
            lowest_mse = gompertz_fitter.mse


    except Exception as e:
        pass
        #print(f"Gompertz Fitting failed due to {e}. Condition: {subset['Condition'].unique()}")

    try:
        # Fit natural logarithm function
        naturalLogFitter = NaturalLogFitter()
        naturalLogFitter.fit(x, y)
        naturalLogFitter.get_fit_metrics(x, y)

        if naturalLogFitter.mse < lowest_mse:
            best_fitter = naturalLogFitter
            lowest_mse = naturalLogFitter.mse

    except Exception as e:
        pass
        #naturalLogFitter = None
        #natural_log_mse = float('inf')
        #print(f"Natural Logarithm Fitting failed due to {e}. Condition: {subset['Condition'].unique()}")



    if best_fitter is None:
        raise ValueError(f"All fitting attempts failed for well: {well}" )



    # get metrics
    if best_fitter is None:
        ## assign values
        subset["max_yield"] = None
        subset["max_rate"] = None
        subset["inflection_point"] = None

    else:

        if best_fitter is asym_fitter:

            # Generate predicted y values using the fitted curve
            x_values = np.linspace(min(x), max(x), 100)
            y_values = best_fitter.predict(x_values)

            # get max RFUs
            max_yield_x, max_yield_y = best_fitter.get_max_yield(x_values)

            # get max rate
            x_max_rate, y_max_rate = best_fitter.get_max_rate(x)

            # get inflection point
            x_inflection, y_inflection = best_fitter.get_inflection_point(x)

            ## assign values
            subset["max_yield"] = max_yield_y
            subset["max_yield_time"] = max_yield_x
            subset["max_rate"] = y_max_rate
            subset["inflection_point"] = x_inflection
            subset["FeatureExtractionFitter"] = "ASymSig"


        elif best_fitter is gompertz_fitter:

            # Generate predicted y values using the fitted curve
            x_values = np.linspace(min(x), max(x), 100)
            y_values = best_fitter.predict(x_values)

            # get max RFUs
            max_yield_x, max_yield_y = best_fitter.get_max_yield(x_values)

            # get max rate
            x_max_rate, y_max_rate = best_fitter.get_max_rate(x)

            # get inflection point
            x_inflection, y_inflection = best_fitter.get_inflection_point(x)

            ## assign values
            subset["max_yield"] = max_yield_y
            subset["max_yield_time"] = max_yield_x
            subset["max_rate"] = y_max_rate
            subset["inflection_point"] = x_inflection
            subset["FeatureExtractionFitter"] = "Gompertz"

        elif best_fitter is naturalLogFitter:
            # Generate predicted y values using the fitted curve
            x_values = np.linspace(min(x), max(x), 100)
            y_values = best_fitter.predict(x_values)

            # get max RFUs
            max_yield_x, max_yield_y = best_fitter.get_max_yield(x_values)

            ## assign values
            subset["max_yield"] = max_yield_y
            subset["max_yield_time"] = max_yield_x
            subset["max_rate"] = None
            subset["inflection_point"] = None
            subset["FeatureExtractionFitter"] = "NatLog"

    # add the df to the new one
    data_with_features = pd.concat([data_with_features, subset])




import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
## Initial trimming.
# drop negative control
plotting_data = data_with_features[data_with_features["Time"] <= 100]

plotting_data = plotting_data[
    [
        'Time',
        'Well',
        'RFUs',
        'GFP_uM',
        'Condition_id',
        "max_yield",
        "max_yield_time"
        ]
    ]

## time course

fig = plt.figure(figsize=(6, 5))
fig.suptitle("Fluorescence of All Conditions Over Time", fontsize=15)



ax = sns.lineplot(
    data=plotting_data,
    y="GFP_uM",
    x="Time",
    hue="Condition_id",
    palette="magma",
    legend=False
)

sns.scatterplot(
    data=plotting_data,
    y="max_yield",
    x="max_yield_time",
    hue="Condition_id",
    palette="magma",
    legend=False,
    ax=ax
)

# Set maximum number of ticks on each axis to 4
ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

# Increase tick label size
ax.tick_params(axis='both', which='major', labelsize=15)  # Adjust the size as needed

# Set new axis labels
ax.set_xlabel("Time (Mins)", fontsize=15)  # New x-axis label with adjusted font size
ax.set_ylabel("Equivalent GFP Fluorescence (uM)", fontsize=15)  # New y-axis label with adjusted font size

# Manually create and set the legend entries
unique_conditions = plotting_data["Condition_id"].unique()
palette = sns.color_palette("magma", len(unique_conditions))
handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', label=condition)
           for condition, color in zip(unique_conditions, palette)]

legend = plt.legend(handles=handles, title="Condition Id", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(legend.get_title(), fontsize=15)  # Set the fontsize of the legend title

plt.tight_layout()

# Save the figure
plt.savefig(project_path + "Datasets/All_Conditions_over_Time.png")
plt.clf()







data_with_features.reset_index(drop=True)

data_with_features.to_csv(
    project_path + "/Datasets/" +
    "tidy_dataset_with_all_time.csv",
    index = None
    )

# drop drop_duplicates
data_with_features = data_with_features.drop(["Time", "RFUs", "GFP_uM"], axis = 1)
data_with_features = data_with_features.drop_duplicates()

data_with_features.to_csv(
    project_path + "/Datasets/" +
    "tidy_dataset.csv",
    index = None
    )


