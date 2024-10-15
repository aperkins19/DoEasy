import sys
import json
import pickle

import pandas as pd
import numpy as np 
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 

from sub_scripts.modelling import generate_feature_matrix, LinearRegressionModel


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

def lookup_variable_type(design_parameters_dict, term):
    ## check if last three elements are "**int"
    if len(term) >= 3 and term[-3:-1] == '**' and term[-1].isdigit():
        term_for_lookup = term[:-3]
    else:
        term_for_lookup = term
    
    ## look up term in design_params_dict to get the variable type e.g. categorifical and the assign the appropriate patsy genre e.g. "C"
    if design_parameters_dict["Variables"][term_for_lookup]["Type"] == "Continuous":
        patsy_genre = "I"
    elif design_parameters_dict["Variables"][term_for_lookup]["Type"] == "Categorical":
        patsy_genre = "C"
    else:
        print("Unknown variable type:", term_for_lookup, " treating as continuous")
        patsy_genre = "I"
    return patsy_genre

def Anova(numeric_coded_design_with_response_values: pd.DataFrame, model_terms: list, response_variables: list, design_parameters_dict: dict):

    # Importing libraries 
    import statsmodels.api as sm 
    from statsmodels.formula.api import ols 

    def generate_formula(model_terms, design_parameters_dict):

        #print(model_terms)

        # initalise formula.
        formula = response_variables[0] + " ~ "

        # iterate over terms
        for term in model_terms:

            
            if term.count(".") == 0:
                 ## look up term in design_params_dict to get the variable type e.g. categorifical and the assign the appropriate patsy genre e.g. "C"
                patsy_genre = lookup_variable_type(design_parameters_dict, term)
                
                # check if term is higher order
                if term.count("*") != 0:

                    "np.power("+term+", "+str(term.count('*'))+")"
                    formula = formula + "np.power("+term[:-3]+", "+str(term.count('*'))+") + "

                else:
                    formula = formula + patsy_genre+"("+term+") + "

            elif term.count(".") >= 1:
                # strip and reformat in to C(v1):C(v2)

                # Split the string by '.'
                termparts = term.split('.')

                # Strip whitespace and format each part
                formatted_terms = []
                for part in termparts:
                    individual_term = part.strip()  # Strip whitespace from the current part
                    ## look up term in design_params_dict to get the variable type e.g. categorifical and the assign the appropriate patsy genre e.g. "C"
                    patsy_genre = lookup_variable_type(design_parameters_dict, individual_term)
                    formatted_term = patsy_genre+"("+individual_term+")"
                    formatted_terms.append(formatted_term)  # Add the formatted part to the list


                # Join the formatted parts with a colon
                result_string = ":".join(formatted_terms)

                formula = formula + result_string + " + "

        # remove last " +"
        formula = formula.rstrip(" + ")

        return formula


    formula = generate_formula(model_terms, design_parameters_dict) 
    print(formula)
    print(numeric_coded_design_with_response_values.dtypes)


    # Performing two-way ANOVA 
    model = ols(
        formula
        , data=coded_design_with_response_values).fit() 
    result_table = sm.stats.anova_lm(model, type=2) 
    result_table=result_table.reset_index()

    return result_table

#import the project path 
project_path = sys.argv[1]
model_path = sys.argv[1]



# import design parameters
design_parameters_dict = json.load(open(project_path + "design_parameters.json", "r"))

# import dataset
tidy_data = pd.read_csv(project_path+"/Datasets/tidy_dataset.csv")

# import coded design
coded_design = pd.read_csv(project_path+"/Experiment_Designs/design.csv")

# import screening_model_config 
screening_model_config_dict = json.load(open(project_path + "screening_model_config.json", "r"))
model_terms = screening_model_config_dict["model_terms"]

# get variable name lists
input_variables = list(design_parameters_dict["Variables"].keys())
response_variables = list(design_parameters_dict["Response_Variables"].keys())

# generate coded design with response values
# Merging coded_design with response variables from tidy_data via condition_id
coded_design_with_response_values = coded_design.merge(tidy_data[["Condition_id"]+response_variables], on="Condition_id", how="right")
# copy design
numeric_coded_design_with_response_values = coded_design_with_response_values.copy()
# replace all "-" and "+" to numeric equivialents 
numeric_coded_design_with_response_values[input_variables] = numeric_coded_design_with_response_values[input_variables].replace({'-': -1, '+': 1})
numeric_coded_design_with_response_values[input_variables] = numeric_coded_design_with_response_values[input_variables].astype(float)


## generate feature matrix
input_matrix = numeric_coded_design_with_response_values[input_variables].copy()

feature_matrix = generate_feature_matrix(input_matrix, model_terms)


model_matrix = feature_matrix.copy()
model_matrix[response_variables] = coded_design_with_response_values[response_variables]


## all response variables
for response_variable in response_variables:


    linear_model = LinearRegressionModel(
        feature_matrix = model_matrix,
        model_terms=model_terms,
        input_variables = input_variables,
        response_variable = response_variable,
        design_parameters_dict = design_parameters_dict,
        model_path = model_path
        )

    linear_model.fit()


    linear_model.assess_fit_with_training_data()
    significance_results = linear_model.model_term_significance_analysis(model_path, dont_plot_intercept=True, for_figure=True)


