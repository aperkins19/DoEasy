### import
import sys
import json
import pickle
import pandas as pd
from sub_scripts.modelling import generate_feature_matrix, LinearRegressionModel

#import the project path 
project_path = sys.argv[1]
model_path = sys.argv[2]
feature_to_model = sys.argv[3]
modeltype = sys.argv[4]


# get model
with open(model_path + "/" + "fitted_model.pkl", 'rb') as file:
    model = pickle.load(file)

# assess fit with training_data

training_results_df = model.assess_fit_with_training_data()


#################### if validation data present, conduct assessment
# import dataset
tidy_data = pd.read_csv(project_path+"/Datasets/tidy_dataset.csv")


# generate coded design with response values
# first get the validation data
if tidy_data['DataPointType'].str.contains('Validation').any():

    # import design parameters
    design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))

    # import coded design
    coded_design = pd.read_csv(project_path+"/Experiment_Designs/design.csv")

    # import model_params_dict 
    model_params_dict = json.load(open(model_path + "model_params.json", "r"))
    model_terms = model_params_dict["model_terms"]

    # get variable name lists
    input_variables = list(design_parameters_dict["Variables"].keys())
    response_variables = list(design_parameters_dict["Response_Variables"].keys())

    validation_data = tidy_data[tidy_data["DataPointType"] == "Validation"].copy()

    input_matrix = validation_data[input_variables].copy()
    feature_matrix = generate_feature_matrix(input_matrix, model_terms)

    # append Y
    feature_matrix[feature_to_model] = tidy_data[feature_to_model]

    validation_results_df = model.assess_fit_with_validation_data(feature_matrix)

    # Merge results
    training_results_df["Validation Data"] = validation_results_df["Validation Data"].copy()


print()
print("Model Fit Assessment: ")
print(training_results_df)
print()
training_results_df.to_csv(model_path + "Model_Fit_Assessment.csv", index=None)
print("Report saved in the model directort as Model_Fit_Assessment.csv")
print()


############# Observations vs Predictions

# build feature matrix containing both training and validation data if applicable

# import design parameters
design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))
# import model_params_dict
model_params_dict = json.load(open(model_path + "model_params.json", "r"))
model_terms = model_params_dict["model_terms"]



input_variables = list(design_parameters_dict["Variables"].keys())
model_terms = model_params_dict["model_terms"]

# regen feature matrix from tidydata
input_matrix = tidy_data[input_variables].copy()
feature_matrix = generate_feature_matrix(input_matrix, model_terms)


# append Y & Datapoint type
feature_matrix[feature_to_model] = tidy_data[feature_to_model]
feature_matrix["DataPointType"] = tidy_data["DataPointType"]

# generate obs vs preds plot
model.observations_vs_predictions(feature_matrix, project_path, model_path)

# save coefficient csv and latex
# Define the model terms and coefficients
model_terms = [term.replace("_", "$\_$") for term in model_terms]
model_terms = [
    term.replace('**', r'\textsuperscript{') + '}' if '**' in term else term
    for term in model_terms
]


model_df = pd.DataFrame(
        {
        "Model Term Symbol": ["$x_" + str(i)+"$" for i, coeff in enumerate(model_terms)],
        "Model Terms": model_terms,
        "Coefficient Symbol": ["$\\beta_" + str(i)+"$" for i, coeff in enumerate(model.model_coefficients)],
        "Coefficients": [round(coeff, 2) for i, coeff in enumerate(model.model_coefficients)]
        }
    )


#11model_df.index = model_df["Symbol"]; model_df.drop("Symbol", axis=1, inplace=True)
model_df.to_csv(model_path + "/" + model.Y_name +"_model_coefficients.csv")

# Convert DataFrame to LaTeX format
latex_code = model_df.style.to_latex()

# Split LaTeX code into lines
latex_lines = latex_code.splitlines()

# Iterate through lines and replace coefficients with rounded values
# Rounded coefficients
rounded_coefficients = [round(coeff, 2) for i, coeff in enumerate(model.model_coefficients)]


updated_latex_code = []
for i, line in enumerate(latex_lines):
    if i > 1:  # Skip header lines
        # Split the line by '&' and check if it has enough elements
        updated_line = line.split('&')
        if len(updated_line) >= 4:
            # Replace the coefficient part with the rounded value
            updated_line[-1] = f" {rounded_coefficients[i-2]:.2f} \\\\"
            updated_latex_code.append(" & ".join(updated_line))
        else:
            updated_latex_code.append(line)  # Leave the line unchanged
    else:
        updated_latex_code.append(line)  # Keep header lines unchanged

# Join the lines back into a LaTeX string
updated_latex_string = "\n".join(updated_latex_code)

# Split LaTeX code into lines
latex_lines = updated_latex_string.splitlines()

# Process each line to remove the first column
updated_latex_code = []
for line in latex_lines:
    if "&" in line:  # Check if it's a data line
        updated_line = line.split("&")[1:]  # Remove the first column (index)
        updated_latex_code.append(" &".join(updated_line))  # Join the remaining columns
    else:
        updated_latex_code.append(line)  # Keep non-data lines unchanged

# Join the lines back into a LaTeX string
updated_latex_string = "\n".join(updated_latex_code)



# Save LaTeX code to a text file
with open(model_path + "/" + model.Y_name +"_model_coefficients.txt", 'w') as f:
    f.write(updated_latex_string)



# save model to persist model stats
with open(model_path + "/" + "fitted_model.pkl", 'wb') as file:
    pickle.dump(model, file)
