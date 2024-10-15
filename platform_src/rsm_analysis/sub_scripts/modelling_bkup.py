import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import sympy as sym
import os
import shutil
from tqdm import tqdm

def generate_formula(model_terms, response_variable, design_parameters_dict):
    
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

    #print(model_terms)

    # initalise formula.
    formula = response_variable + " ~ "

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

def generate_feature_matrix(input_matrix, model_features):


    feature_matrix = input_matrix.copy()

    # Generate features
    for feature in model_features:

        if feature == "Intercept":
            feature_matrix[f'{feature}'] = 1

        elif '.' in feature:  # Interaction terms
            factors = feature.split('.')
            # Calculate the product of the factors
            interaction_term = feature_matrix[factors[0]].copy()
            for factor in factors[1:]:
                interaction_term *= feature_matrix[factor].copy()
            feature_matrix[f'{feature}'] = interaction_term

        elif '**' in feature:  # Polynomial terms
            factor, power = feature.split('**')
            feature_matrix[f'{feature}'] = feature_matrix[factor].copy() ** int(power)

    return feature_matrix


#def generate_model_matrix():
#    return 

def LinearRegression_Model(df, input_variables, model_features, response_variable, design_parameters_dict, Anova_Type:int):

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import pandas as pd


    # Assuming df is your DataFrame and y is your target variable
    X = df[model_features]
    y = df[response_variable]  # Your response variable

    # define and fit model
    model = LinearRegression().fit(X, y)
    print(model.coef_)
    print(model.intercept_)
    
    # get y preds
    y_Preds = model.predict(X)

    n_datapoints = X.shape[0] * X.shape[1]

    # calculate the summary stats of model.
    grand_mean = y.mean()
    # global anova stats
    sum_of_squares_total = np.sum((y - grand_mean)**2) # aka:TSS the variability inherent in the dependent variable.
    sum_of_squares_of_model = np.sum([(y_hat_i - grand_mean)**2 for y_hat_i in y_Preds]) # aka regression sum squares RSS,  represents the variability that the model explains. nadanai has a factor of two here...
    residual_sum_squares = np.sum((y - y_Preds)**2) # variability that the model does not explain.
    print()
    print("ssm + ssmpe", sum_of_squares_of_model+residual_sum_squares)
    print("sum_of_squares_total", sum_of_squares_total)
    print()

    mean_squared_error = np.mean((y - y_Preds)**2)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    # stats
    r2 = 1 - (residual_sum_squares/sum_of_squares_total) # r2
    adjusted_r2 = 1 - (1 - r2) * ( (X.shape[0] - 1) / (X.shape[0] - X.shape[1] - 1) ) # The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. It is particularly useful in multiple regression models, as it compensates for the addition of variables that do not improve the model. 
   
    # degrees of freedom
    #df_total = X.shape[0] - 1
    #df_model = len(y_Preds) - 1
    dof = 2

   
    # ANOVA - model 
    mean_square_of_model = sum_of_squares_of_model/(dof)
    mean_square_of_pure_error = residual_sum_squares/(n_datapoints-(dof+1))
    total_mean_square = sum_of_squares_total/(n_datapoints-1)
    F_Statistic = mean_square_of_model/mean_square_of_pure_error

    print("sum_of_squares_of_model", sum_of_squares_of_model)
    print("mean_square_of_pure_error", mean_square_of_pure_error)
    print("sum_of_squares_total", sum_of_squares_total)

    print("mean_square_of_model", mean_square_of_model)
    print("mean_square_of_pure_error", mean_square_of_pure_error)
    print("total_mean_square", total_mean_square)
    print("F_Statistic", F_Statistic)
    print("")

    # degrees of freedom = number of model features



def ANOVA_Model(df, model_features, response_variable, design_parameters_dict, Anova_Type:int):

    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # generate formula
    formula = generate_formula(model_features, response_variable, design_parameters_dict)
    # define model with formula
    model = smf.ols(formula, data=df).fit()
    # model summary
    print(model.summary()) 
    # Perform ANOVA
    anova_results = anova_lm(model, typ=Anova_Type)  # typ=1 for Type I ANOVA
    print(anova_results)



import json
from sklearn.linear_model import LinearRegression

class LinearRegressionModel:

    def __init__(
        self,
        feature_matrix,
        model_terms,
        input_variables,
        response_variable,
        design_parameters_dict,
        model_path
        ):
        self.feature_matrix = feature_matrix
        self.X_true = self.feature_matrix[input_variables].copy()
        self.X_true_feature_matrix = self.feature_matrix[model_terms].copy()
        self.Y_true = self.feature_matrix[response_variable].copy()
        
        self.X_names = input_variables
        self.model_terms = model_terms
        self.Y_name = response_variable
        self.model_response_variable = response_variable
        
        self.model_object = LinearRegression()

        self.model_path = model_path

    # def fit(self):
    #     # define and fit model
    #     self.model_object.fit(self.X_true_feature_matrix, self.Y_true)

    def fit(self):
        X = self.X_true_feature_matrix
        Y = self.Y_true
        # coefficients = inverse(X.T @ X) @ (X.T @ Y)
        self.model_coefficients = np.linalg.inv(X.T @ X) @ (X.T @ Y)

    def predict(self, X_new):

        if not hasattr(self, 'model_coefficients'):
            raise Exception("Model not yet fitted, cannot produce predictions.")
        else:
            # Add a column of ones
            X_new = np.c_[np.ones(X_new.shape[0]), X_new]
            return X_new @ self.model_coefficients



    def assess_fit_with_training_data(self):

        # get y preds
        self.Y_Predicted = self.model_object.predict(self.X_true_feature_matrix)

        # calculate the summary stats of model.
        self.grand_mean = self.Y_true.mean()
        # global anova stats
        self.sum_of_squares_total = np.sum((self.Y_true - self.grand_mean)**2) # aka:TSS the variability inherent in the dependent variable.
        self.sum_of_squares_due_to_regression = np.sum([(y_hat_i - self.grand_mean)**2 for y_hat_i in self.Y_Predicted]) # aka summ of squares due to regression SSR,  represents the variability that the model explains.
        self.residual_sum_squares = np.sum((self.Y_true - self.Y_Predicted)**2) # variability that the model does not explain.

        self.mean_squared_error = np.mean((self.Y_true - self.Y_Predicted)**2)
        self.root_mean_squared_error = np.sqrt(self.mean_squared_error)

        # stats ########### Do i use feature matrix or 
        self.r2 = 1 - (self.residual_sum_squares/self.sum_of_squares_total) # r2
        self.adjusted_r2 = 1 - (1 - self.r2) * ( (self.X_true.shape[0] - 1) / (self.X_true.shape[0] - self.X_true.shape[1] - 1) ) # The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. It is particularly useful in multiple regression models, as it compensates for the addition of variables that do not improve the model. 

        return  pd.DataFrame(
                    [
                    ("Y Grand Mean", self.grand_mean),
                    ("Sum of Squares Total", self.sum_of_squares_total),
                    ("Sum of Squares Model", self.sum_of_squares_due_to_regression),
                    ("Sum of Squares Residual", self.residual_sum_squares),
                    ("Mean Squared Error", self.mean_squared_error),
                    ("Root Mean Squared Error", self.root_mean_squared_error),
                    ("R2", self.r2),
                    ("Adjusted R2", self.adjusted_r2)
                    ],
                    columns = ["Stat", "Training Data"]
                    )

    
    def assess_fit_with_validation_data(self, validation_data_feature_matrix):


        # extract validation
        self.X_validation_true = validation_data_feature_matrix[self.X_names].copy()
        self.X_validation_true_feature_matrix = validation_data_feature_matrix[self.model_terms].copy()
        self.Y_validation_true = validation_data_feature_matrix[self.model_response_variable].copy()

        # predict X_validation
        # get y preds
        self.Validation_Y_Predicted = self.model_object.predict(self.X_validation_true_feature_matrix)

        # calculate the summary stats of model.
        self.Validation_Y_grand_mean = self.Y_validation_true.mean()
        # global anova stats
        self.Validation_sum_of_squares_total = np.sum((self.Y_validation_true - self.Validation_Y_grand_mean)**2) # aka:TSS the variability inherent in the dependent variable.
        self.Validation_sum_of_squares_due_to_regression = np.sum([(y_hat_i - self.Validation_Y_grand_mean)**2 for y_hat_i in self.Validation_Y_Predicted]) # aka regression sum squares RSS,  represents the variability that the model explains. nadanai has a factor of two here...
        self.Validation_residual_sum_squares = np.sum((self.Y_validation_true - self.Validation_Y_Predicted)**2) # variability that the model does not explain.
        self.Validation_mean_squared_error = np.mean((self.Y_validation_true - self.Validation_Y_Predicted)**2)
        self.Validation_root_mean_squared_error = np.sqrt(self.Validation_mean_squared_error)
        
        # stats
        self.Validation_r2 = 1 - (self.Validation_residual_sum_squares/self.Validation_sum_of_squares_total) # r2
        self.Validation_adjusted_r2 = 1 - (1 - self.Validation_r2) * ( (self.X_validation_true.shape[0] - 1) / (self.X_validation_true.shape[0] - self.X_validation_true.shape[1] - 1) ) # The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. It is particularly useful in multiple regression models, as it compensates for the addition of variables that do not improve the model. 

        return  pd.DataFrame(
                [
                ("Y Grand Mean", self.Validation_Y_grand_mean),
                ("Sum of Squares Total", self.Validation_sum_of_squares_total),
                ("Sum of Squares Model", self.Validation_sum_of_squares_due_to_regression),
                ("Sum of Squares Residual", self.Validation_residual_sum_squares),
                ("Mean Squared Error", self.Validation_mean_squared_error),
                ("Root Mean Squared Error", self.Validation_root_mean_squared_error),
                ("R2", self.Validation_r2),
                ("Adjusted R2", self.Validation_adjusted_r2)
                ],
                columns = ["Stat", "Validation Data"]
                )

    def observations_vs_predictions(self, feature_matrix, project_path, model_path):
        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")

        # generate DF with observations, preds and append Datapoint type
        performance_df = pd.DataFrame()
        performance_df["Observations"] = feature_matrix[self.model_response_variable].copy()
        performance_df["Predictions"] = self.model_object.predict(feature_matrix[self.model_terms])
        performance_df["DataPointType"] = feature_matrix["DataPointType"].copy()

        # determine if Validation for plotting colour coding
        performance_df["IsValidation"] = np.where(performance_df["DataPointType"] == "Validation", "Validation", "Training")

        # generate residuals
        performance_df["Residuals"] = performance_df["Observations"] - performance_df["Predictions"]
        performance_df["Zeros"] = 0

        # plotting

        fig, ax = plt.subplots(nrows=1, ncols=2)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        # set datapoint colours
        IsValidationColours = {"Validation": "red", "Training": "blue"}


        fig.suptitle("Model Performance", fontsize=18)

        # observations vs predictions
        ax[0].set_title("Observed vs Predicted")

        # begin plot
        sns.scatterplot(
            data = performance_df,
            x = "Predictions",
            y = "Observations",
            hue = "IsValidation",
            palette = IsValidationColours,
            ax = ax[0]
            )
                    
        # y = x
        sns.lineplot(
            data = performance_df,
            x = "Observations",
            y = "Observations",
            linestyle="--",
            color = "red",
            ax = ax[0]
        )

        # residuals plot
        ax[1].set_title("Residuals")

        sns.scatterplot(
            data = performance_df,
            x = "Observations",
            y = "Residuals",
            hue = "IsValidation",
            palette = IsValidationColours,
            ax = ax[1]
            )
        # y = x
        sns.lineplot(
            data = performance_df,
            x = "Observations",
            y = "Zeros",
            color = "red",
            linestyle="--",
            ax = ax[1]
        )

        fig.tight_layout()            

        plt.savefig(model_path + "/model_truth_vs_preds.png")

        plt.clf()

        return performance_df



    def ols_stats(self):
        import statsmodels.api as sm

        # Generate some random data


        # Convert X to a DataFrame and add an intercept
        X_df = self.X_true_feature_matrix.copy()
        X_df = sm.add_constant(X_df)

        # Fit the OLS model
        model = sm.OLS(self.Y_true, X_df).fit()

        # Print the summary
        print(model.summary())


    # compare with ols
    # check intercept

    def model_term_significance_analysis(self, model_path):

        import scipy.stats as stats

        # Example coefficients and standard errors
        model_terms = self.model_terms
        coefficients = self.model_object.coef_  # Example coefficients


        n = self.X_true.shape[0]  # Number of observations
        p = len(self.model_terms)  # Number of parameters (including intercept if applicable)
        rss = self.residual_sum_squares  # Your calculation of RSS

        dof = n - p

        estimated_variance_of_error_term = rss / dof

        # this is the matrix of the feature matrix in np form.
        feature_matrix_matrix = self.X_true_feature_matrix.values
        # Calculate (X'X)^(-1)
        # first the matrix is transposed
        # the matrix multipled by feature matrix this produces the square matrix (cross-products of independent variables)
        # the square matrix is used to generate the inverse square matrix
        # the inverse square matrix 
        XX_inv = np.linalg.inv(feature_matrix_matrix.T @ feature_matrix_matrix)
        # Multiply by estimated variance of the error term
        var_covar_matrix = estimated_variance_of_error_term * XX_inv
        # The diagonal of this matrix contains the variances of the estimates
        var_estimates = np.diag(var_covar_matrix)
        # Standard errors are the square roots of these variances
        standard_errors = np.sqrt(var_estimates)
        print("Standard Errors of the Coefficients:", standard_errors)

        # Calculate t-statistics for each coefficient
        t_statistics = coefficients / standard_errors

        # Calculate the p-values for each t-statistic
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=dof)) for t in t_statistics]

        significance_results = pd.DataFrame()
        significance_results["Model Terms"] = self.model_terms
        significance_results["Coefficient"] = coefficients
        significance_results["T-Statistic"] = t_statistics
        significance_results["p-Value"] = p_values

        print(significance_results)
        print()
        print("intercept")
        print(self.model_object.intercept_)


        # Plot
        # Create a figure with subplots
        fig, ax = plt.subplots(nrows=1, ncols=3)
        fig.set_figheight(5)
        fig.set_figwidth(15)

        ax[0].set_title("Coefficient")

        # begin plot
        sns.barplot(
            data = significance_results,
            x = "Coefficient",
            y = "Model Terms",
            ax = ax[0]
            )
  
        # residuals plot
        ax[1].set_title("T-Statistic")

        # begin plot
        sns.barplot(
            data = significance_results,
            x = "T-Statistic",
            y = "Model Terms",
            ax = ax[1]
            )

        # residuals plot
        ax[2].set_title("p-Value")

        # begin plot
        sns.barplot(
            data = significance_results,
            x = "p-Value",
            y = "Model Terms",
            ax = ax[2]
            )
            
        fig.tight_layout()            

        plt.savefig(model_path + "/model_terms_signficance.png")
        plt.clf()





    def t_distribution_visualised(self, model_path):
        from scipy.stats import t, norm, cauchy

        dofs = np.arange(1,5,1)
        x = np.arange(-10,10, 0.01)
        print(cauchy.pdf(x))
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(10,10))

        # begin plot
        ax = fig.add_subplot(111)

        ax.plot(x, norm.pdf(x), label="Normal", alpha=0.6)
        ax.plot(x, cauchy.pdf(x), label="Cauchy", alpha=0.6)

        for dof in dofs:
            print(dof)
            p = t.pdf(np.abs(x), dof)
            ax.plot(x, p, label = dof, alpha=0.4, linestyle="dotted")

        ax.plot(x, t.pdf(np.abs(x), 30), label = 30, alpha=0.6, linestyle="--")
        
        ax.legend()
        fig.suptitle("PDF of Student's T-Distribution with varying degrees of freedom.")
        fig.savefig(model_path+"/t_dist.png")





    def find_max_y(
        self,
        model_path,
        replicates,
        decimal_point_threshold = 1,
        deep_search = False
        ):

        # Simulated annealing
        from sub_scripts.optimisation import simulated_annealing_multivariate
        from sub_scripts.optimisation import simple_greedy_search_multi
        import random
        
        ## define search space
        ordered_data = self.X_true[self.X_names]
        # Finding min and max values for each column
        min_values = ordered_data.min()
        max_values = ordered_data.max()
        # Creating a list of tuples (min, max) for each column
        min_max_pairs = [(min_values[var], max_values[var]) for var in ordered_data.columns]

        SA_Hyper_Params = {
            "Temperature": 5,
            "Cooling_Schedule": 0.95,
            "N_Proposals": 120
        }


        # initialise solutions lists
        max_x_list = []
        max_y_list = []



        if deep_search:

            print()
            print("Conducting Deep Search..")
            print(f"First, the StdDev of {replicates} samples will be calculated.")
            print(f"Then further samples will be taken one at a time and the StdDev of the most recent {replicates} recalculated.")
            print(f"This will repeat until until the StdDev matches the StdDev of the previous iteration to a threshold of {decimal_point_threshold} decimal places.")


            # conduct initial sampling
            for i in range(replicates):
                print("Sampling Chain: " + str(i)+"/"+str(replicates))
                final_max_x, final_max_y, history_x = simulated_annealing_multivariate(
                    model = self.model_object,
                    search_space_real = min_max_pairs,
                    data_feature_generator = generate_feature_matrix,
                    data_feature_generator_args = [self.X_names, self.model_terms, self.X_true_feature_matrix],
                    show_progress_bar = True,
                    SA_Hyper_Params = SA_Hyper_Params
                    )
                # concat to list
                max_x_list.append(final_max_x)
                max_y_list.append(final_max_y)

            #calculate the rounded stddev
            avg_std_current = round(np.std(np.concatenate(max_y_list)[-replicates:]), decimal_point_threshold)
            print()
            print("avg_std_current:", avg_std_current)
            # init avg_std_prop
            avg_std_prop = np.nan

            # if the rolling std doesn't match the current std - repeat
            while avg_std_current != avg_std_prop:
                
                # update current
                avg_std_current = avg_std_prop
                # resample
                final_max_x, final_max_y, history_x = simulated_annealing_multivariate(
                    model = self.model_object,
                    search_space_real = min_max_pairs,
                    data_feature_generator = generate_feature_matrix,
                    data_feature_generator_args = [self.X_names, self.model_terms, self.X_true_feature_matrix],
                    show_progress_bar = True,
                    SA_Hyper_Params = SA_Hyper_Params
                    )
                # add new solutions to lists
                max_x_list.append(final_max_x)
                max_y_list.append(final_max_y)

                # recalculate the avg_std_prop with the updated list
                avg_std_prop = round(np.std(np.concatenate(max_y_list)[-replicates:]), decimal_point_threshold)

                print()
                print("avg_std_current:", avg_std_current)
                print("avg_std_prop:", avg_std_prop)
                print()

        else:
            print()
            print("Conducting Simple Search..")
            print(f"The StdDev of {replicates} samples will be calculated.")

            for i in range(1, (replicates+1),1):

                print("Sampling Chain: " + str(i)+"/"+str(replicates))
                final_max_x, final_max_y, history_x = simulated_annealing_multivariate(
                        model = self.model_object,
                        search_space_real = min_max_pairs,
                        data_feature_generator = generate_feature_matrix,
                        data_feature_generator_args = [self.X_names, self.model_terms, self.X_true_feature_matrix],
                        show_progress_bar = True,
                        SA_Hyper_Params = SA_Hyper_Params
                        )

                max_x_list.append(final_max_x)
                max_y_list.append(final_max_y)


        # Stack the arrays vertically
        max_x_list = np.vstack(max_x_list)
        max_y_list = np.vstack(max_y_list)

        # package in to df and append y column
        max_y_samples = pd.DataFrame(
            max_x_list,
            columns=self.X_names
            )
        max_y_samples[self.Y_name] = max_y_list

        # melt, produce metrics and repivot
        max_y_samples = max_y_samples.melt(var_name="factor", value_name="value")
        # Adding a column with the mean of 'value' for each 'factor'
        max_y_samples['mean'] = max_y_samples.groupby('factor')['value'].transform('mean')
        max_y_samples['stdev'] = max_y_samples.groupby('factor')['value'].transform('std')
        max_y_samples['Num_Samples'] = len(max_x_list)

        # drop duplicates and transpose to produce final table.
        max_y_samples.drop("value", axis=1, inplace=True)
        max_y_samples = max_y_samples.drop_duplicates().set_index("factor").T

        # get scaled distance from middle of design space
        average_distance = pd.DataFrame((max_y_samples.loc["mean"] / pd.concat([self.X_true, self.Y_true], axis=1).mean(axis=0))).T
        average_distance.index = ["Avg_dist"]
        max_y_samples = pd.concat([max_y_samples, average_distance], axis=0)

        max_y_samples = max_y_samples.round(decimal_point_threshold)
        
        # save
        max_y_samples.to_csv(model_path +"/predicted_optimum.csv")

        return max_y_samples



    def generate_x_y_z_prediction_meshes(
        self,
        fixed_level,
        fixed_variable_name,
        fixed_variables_dict,
        variables_compared_names,
        variables_compared_dict
        ):


        # build the mesh
        fixed_mesh = np.full_like(
                variables_compared_dict[variables_compared_names[0]]["mesh"],
                fill_value = fixed_level
                )

        # assign an array of min and max from variable data
        fixed_variables_dict[fixed_variable_name] = {
            "mesh" : fixed_mesh,
            "mesh_flat" : fixed_mesh.flatten()
        }      

        # Create input data
        # combine both dictionaries
        all_variables_dict = {**variables_compared_dict, **fixed_variables_dict}
        # sort to match the order of the variables in the model.
        all_variables_dict = {key: all_variables_dict[key] for key in self.X_names}
        # unpack the flattened arrays and build a tuple
        flattened_grids = tuple([variable_dict["mesh_flat"] for variable_name, variable_dict in all_variables_dict.items()])
        # stack the flattened arrays columnwise
        input_data = np.column_stack(flattened_grids)

        # put in df to give feature names
        input_data = pd.DataFrame(input_data, columns = self.X_names)

        ## generate feature matrix
        feature_matrix = generate_feature_matrix(input_data, self.model_terms)
        # Reorder feature_matrix columns to match X_true_feature_matrix - the training data
        feature_matrix = feature_matrix[list(self.X_true_feature_matrix.columns)]

        # Predict the outputs using the trained model
        predicted_output = self.model_object.predict(feature_matrix)


        # Reshape the predicted outputs for contour plotting
        predicted_output = predicted_output.reshape(
            all_variables_dict[variables_compared_names[0]]["mesh"].shape
            )

        return predicted_output, all_variables_dict





    def get_surface_plots_as_figure(
        self,
        experiment_description,
        model_path,
        feature_to_model,
        plot_ground_truth = False
        ):

        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        if len(self.X_names) == 1:
            raise ValueError("Only one X variable, surface plot not applicable.")

        elif len(self.X_names) == 2:
            raise ValueError("Single surface plot to be implemented.")

        elif len(self.X_names) >= 3:

            variables = list(itertools.combinations(self.X_names, r=2))

            # as the plots are build, save them for figure construction after the loop.
            number_of_combinations = len(variables)

            # Define your desired figsize (width, height in inches)
            figsize = (10, 2 * number_of_combinations)

            is_odd = (number_of_combinations % 2) != 0
            if not is_odd:
                # Create a figure with subplots
                fig, axes = plt.subplots(int(number_of_combinations/2), 2, figsize=figsize, subplot_kw=dict(projection='3d'))  # 2x2 grid for subplots

            # Create a figure with subplots
            fig, axes = plt.subplots(round(number_of_combinations/2), 2, figsize=figsize, subplot_kw=dict(projection='3d')) 
            

            ### mesh generation for each subplot.
            for combination, ax in zip(variables, axes.flat):
                
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared
                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data - real data not the feature matrices
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 100)
                        }
                    variables_compared[variable] = dict_for_assignment


                # get variable_meshes
                variable_arrays = [variable_dict["initial_array"] for variable_name, variable_dict in variables_compared.items()]
                variable_meshes = np.meshgrid(variable_arrays[0], variable_arrays[1])

                # assign the mesh to the appropriate variable
                for variable, mesh in zip(variables_compared, variable_meshes):

                    variables_compared[variable] = {
                        "mesh" : mesh,
                        "mesh_flat" : mesh.flatten()
                    }

                # directly create the fixed meshes
                for variable in variables_fixed:   

                    fixed_mesh = np.full_like(
                            variable_meshes[0],
                            fill_value = np.median(np.sort(self.X_true[variable].unique()))
                            )
    
                    # assign an array of min and max from variable data
                    variables_fixed[variable] = {
                        "mesh" : fixed_mesh,
                        "mesh_flat" : fixed_mesh.flatten()
                    }      

                # Create input data
                # combine both dictionaries
                all_variables_dict = {**variables_compared, **variables_fixed}
                # sort to match the order of the variables in the model.
                all_variables_dict = {key: all_variables_dict[key] for key in self.X_names}
                # unpack the flattened arrays and build a tuple
                flattened_grids = tuple([variable_dict["mesh_flat"] for variable_name, variable_dict in all_variables_dict.items()])
                # stack the flattened arrays columnwise
                input_data = np.column_stack(flattened_grids)

                # put in df to give feature names
                input_data = pd.DataFrame(input_data, columns = self.X_names)

                ## generate feature matrix
                feature_matrix = generate_feature_matrix(input_data, self.model_terms)
                # Reorder feature_matrix columns to match X_true_feature_matrix - the training data
                feature_matrix = feature_matrix[list(self.X_true_feature_matrix.columns)]

                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(feature_matrix)

                # get the variable component names
                variables_compared_names = list(combination)

                # Reshape the predicted outputs for contour plotting
                predicted_output = predicted_output.reshape(
                    all_variables_dict[variables_compared_names[0]]["mesh"].shape
                    )


                # begin plot
                ax = fig.add_subplot(ax, projection="3d")

                # plot model
                ax.plot_surface(
                    all_variables_dict[variables_compared_names[0]]["mesh"],
                    all_variables_dict[variables_compared_names[1]]["mesh"],
                    predicted_output,
                    cmap="plasma"
                    )

                # plot ground truth

                if plot_ground_truth:

                    # grab the values for slicing
                    # import model_config 
                    model_config_dict = json.load(open(model_path + "/model_config.json", "r"))
                    slicing_values_dict = model_config_dict["Data_Slicing_Values_For_Plotting"]


                    # combine real x with real y
                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                    # slice data to get the data points where the fixed variable is the same.                    
                    real_data_slice = full_real_data.copy()
                    # iterate over the fixed variables slicing the data to the mid points
                    for fixed_variable in variables_fixed.keys():

                        real_data_slice = real_data_slice[
                            real_data_slice[fixed_variable] == slicing_values_dict[fixed_variable]
                        ]


                    # make sure x and y are the same as the model grid
                    x_real_data = real_data_slice[variables_compared_names[0]]
                    y_real_data = real_data_slice[variables_compared_names[1]]
                    # extract the y data
                    z_real_data = real_data_slice[self.Y_name]
                    

                    #plot
                    ax.scatter(x_real_data, y_real_data, z_real_data)

                ax.view_init(azim = 30, elev = 5)

                # Create the surface plot
                ax.set_xlabel(variables_compared_names[0])
                ax.set_ylabel(variables_compared_names[1])
                ax.set_zlabel(self.Y_name)
                ax.set_title(
                    variables_compared_names[0]+" vs "+variables_compared_names[1],
                    loc="center")


            # Remove empty subplots if necessary
            if is_odd:
                fig.delaxes(axes.flat[-1])

            fig.suptitle("Surface Plots of Model Fit: " + experiment_description, fontsize= 15)

            plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
            plt.savefig(model_path + "surface_plots/" +  "surface_plot_figure_"+experiment_description+".png")
            plt.clf()
                
        
        else:
            raise ValueError("Shape of X Data unknown.")


    def get_predicted_Y_for_all_level_combinations(self, model_path):

        # import model config to get unique levels
        model_config_dict = json.load(open(model_path + "model_config.json", "r"))
        unique_levels_dict = model_config_dict["Unique_Levels"]

        # Variable names extracted from the keys of the dictionary
        variable_names = list(unique_levels_dict.keys())

        # Updated code to generate meshes using the levels from the dictionary
        df_list = []

        # Generate meshes for each combination of 2 input variables
        # where the third variable is fixed at one of its levels
        for fixed_var_idx, fixed_var in enumerate(variable_names):

            # Determine the other two variables to create a mesh
            other_vars = variable_names[:fixed_var_idx] + variable_names[fixed_var_idx+1:]

            for fixed_level in unique_levels_dict[fixed_var]:

                for i, xvar in enumerate(other_vars):

                    for yvar in other_vars[i+1:]:

                        X, Y = np.meshgrid(unique_levels_dict[xvar], unique_levels_dict[yvar])

                        # Flatten the mesh grids
                        flat_X = X.ravel()
                        flat_Y = Y.ravel()

                        # Create a DataFrame for the mesh
                        df_mesh = pd.DataFrame({
                            xvar: flat_X,
                            yvar: flat_Y,
                            fixed_var: np.full(flat_X.shape, float(fixed_level))  # Repeat the fixed level for all rows
                        })

                        # Append the DataFrame to our list
                        df_list.append(df_mesh)

        # Concatenate all individual DataFrames into one
        df_combinations = pd.concat(df_list, ignore_index=True)
        df_combinations.drop_duplicates()
        ## generate feature matrix
        feature_matrix = generate_feature_matrix(df_combinations, self.model_terms)
        # Reorder feature_matrix columns to match X_true_feature_matrix - the training data
        feature_matrix = feature_matrix[list(self.X_true_feature_matrix.columns)]

        # Predict the outputs using the trained model
        df_combinations[self.Y_name] = self.model_object.predict(feature_matrix)

        return df_combinations




    def get_all_surface_plots(
        self,
        experiment_description,
        model_path,
        project_path,
        plot_args
        ):
            
        
        ## first create the directories for the individual surface plots
        # Extract the array and iterate through it to create dir
        for input_var in self.X_names:
            dir_path = os.path.join(model_path, "surface_plots", input_var)
            # Check if the directory exists
            if os.path.exists(dir_path):
                # If it does, delete it recursively
                shutil.rmtree(dir_path)
            # Now, create the directory
            os.makedirs(dir_path)

        # import design parameters
        design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))


        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        if len(self.X_names) == 1:
            raise ValueError("Only one X variable, surface plot not applicable.")

        elif len(self.X_names) == 2:
            raise ValueError("Single surface plot to be implemented.")

        elif len(self.X_names) >= 3:

            # import model config to get unique levels
            model_config_dict = json.load(open(model_path + "model_config.json", "r"))
            unique_levels_dict = model_config_dict["Unique_Levels"]

            variables = list(itertools.combinations(self.X_names, r=2))
            
            # set the z axis limit
            df_combinations = self.get_predicted_Y_for_all_level_combinations(model_path)

            z_axis_max = df_combinations[self.Y_name].max() * 1.2
            z_axis_min = df_combinations[self.Y_name].min() * 0.8

            ### mesh generation for each subplot.
            for combination in tqdm(variables, desc="Generating Plots"):

                                    
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared

                # get unique levels of combination
                # Filter unique_levels_dict to include only keys that are in variables_fixed
                variables_fixed_unique_levels_dict = {key: unique_levels_dict[key] for key in variables_fixed if key in unique_levels_dict}

                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data - real data not the feature matrices
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 100)
                        }
                    variables_compared[variable] = dict_for_assignment

                
                # get variable_meshes
                variable_arrays = [variable_dict["initial_array"] for variable_name, variable_dict in variables_compared.items()]
                variable_meshes = np.meshgrid(variable_arrays[0], variable_arrays[1])

                # assign the mesh to the appropriate variable
                for variable, mesh in zip(variables_compared, variable_meshes):

                    variables_compared[variable] = {
                        "mesh" : mesh,
                        "mesh_flat" : mesh.flatten()
                    }

                # get the variable component names
                variables_compared_names = list(combination)

                ##### this is where we iterate over the fixed variables levels to generate each subplot.
                # in order to create every possible slice, we fix one variable whilst we iterate over the over.
                # this one fixes the root variable


                for variable in variables_fixed:
                    # we then iterate again to get the "working var"
                    for working_var in variables_fixed:
                        # if they match, skip
                        if (variable == working_var) and (len(variables_fixed) != 1):
                            pass 
                        else:
                            
                            #iterate over the unique levels of the working var
                            for i, level in enumerate(variables_fixed_unique_levels_dict[working_var]):
                                
                                # generate the prediction meshes
                                predicted_output, all_variables_dict = self.generate_x_y_z_prediction_meshes(
                                        fixed_level = level,
                                        fixed_variable_name = variable,
                                        fixed_variables_dict = variables_fixed,
                                        variables_compared_names = variables_compared_names,
                                        variables_compared_dict = variables_compared                                        )



                                # Create the surface plot

                                # Create a figure with subplots
                                fig = plt.figure(figsize=(
                                    plot_args["Surface"]["Fig_Size_x"],
                                    plot_args["Surface"]["Fig_Size_y"]
                                    )
                                    )

                                # begin plot
                                ax = fig.add_subplot(111, projection="3d")

                                # axis limits
                                ax.set_xlim(
                                    xmin = min(variables_compared[variables_compared_names[0]]["mesh_flat"]),
                                    xmax = max(variables_compared[variables_compared_names[0]]["mesh_flat"]),
                                    )
                                ax.set_ylim(
                                    ymin = min(variables_compared[variables_compared_names[1]]["mesh_flat"]),
                                    ymax = max(variables_compared[variables_compared_names[1]]["mesh_flat"]),
                                    )
                                ax.set_zlim(
                                    zmin = z_axis_min,
                                    zmax = z_axis_max
                                    )

                                # axis labels
                                ax.set_xlabel(variables_compared_names[0], fontsize = plot_args["Surface"]["Axis_label_font_size"])
                                ax.set_ylabel(variables_compared_names[1], fontsize = plot_args["Surface"]["Axis_label_font_size"])
                                ax.set_zlabel(self.Y_name, fontsize = plot_args["Surface"]["Axis_label_font_size"])

                                # light source
                                # Create a light source coming from the left top
                                from matplotlib.colors import LightSource
                                light = LightSource(45, 45)

                                # Set the viewpoint
                                if plot_args["Surface"]["plot_zero_surface"]:
                                    elevation = plot_args["Surface"]["view_zero_surface_elevation"]  # Degrees from the x-y plane
                                    azimuth = plot_args["Surface"]["view_zero_surface_azimuth"]    # Degrees around the z-axis
                                else:
                                    elevation = plot_args["Surface"]["view_normal_elevation"]  # Degrees from the x-y plane
                                    azimuth = plot_args["Surface"]["view_normal_azimuth"]    # Degrees around the z-axis   

                                ax.view_init(elev=elevation, azim=azimuth)

                                if plot_args["Surface"]["plot_zero_surface"]:

                                    ### segregate data sets 
                                    ## upper
                                    # Filter the meshes to keep only the values where predicted_output > 0
                                    mask_upper = predicted_output > 0
                                
                                    # Copy the arrays to avoid changing the original data
                                    x_upper = np.copy(all_variables_dict[variables_compared_names[0]]["mesh"])
                                    y_upper = np.copy(all_variables_dict[variables_compared_names[1]]["mesh"])
                                    z_upper = np.copy(predicted_output)

                                    # Apply the mask, setting values where predicted_output is not greater than 0 to np.nan
                                    x_upper[~mask_upper] = np.nan
                                    y_upper[~mask_upper] = np.nan
                                    z_upper[~mask_upper] = np.nan

                                    ## lower
                                    # Filter the meshes to keep only the values where predicted_output > 0
                                    mask_lower = predicted_output < 0
                                
                                    # Copy the arrays to avoid changing the original data
                                    x_lower = np.copy(all_variables_dict[variables_compared_names[0]]["mesh"])
                                    y_lower = np.copy(all_variables_dict[variables_compared_names[1]]["mesh"])
                                    z_lower = np.copy(predicted_output)

                                    # Apply the mask, setting values where predicted_output is not greater than 0 to np.nan
                                    x_lower[~mask_lower] = np.nan
                                    y_lower[~mask_lower] = np.nan
                                    z_lower[~mask_lower] = np.nan
                                    
                                    # Define kwargs for the model plots
                                    lower_surface_kwargs = {
                                        "rstride": 1,
                                        "cstride": 1,
                                        "cmap": 'plasma',
                                        "linewidth": 0,
                                        "antialiased": True,
                                        "alpha": 0.5,  # Set transparency so we can see the overlap
                                        "zorder": 0
                                    }
                                    # Plot the first surface
                                    ax.plot_surface(
                                        x_lower,
                                        y_lower,
                                        z_lower,
                                            **lower_surface_kwargs
                                            )

                                    # Define kwargs for the plane method
                                    plane_kwargs = {
                                        "rstride": 1,
                                        "cstride": 1,
                                        "linewidth": 0,
                                        "antialiased": True,
                                        "alpha": 0.5,  # Set transparency so we can see the overlap
                                        "zorder": 1,
                                        "color": "grey"
                                    }
                                    # Plot the plane at z=0 with a higher zorder so it's drawn last
                                    ax.plot_surface(
                                        all_variables_dict[variables_compared_names[0]]["mesh"],
                                        all_variables_dict[variables_compared_names[1]]["mesh"],
                                        np.zeros_like(all_variables_dict[variables_compared_names[0]]["mesh"]),
                                        **plane_kwargs
                                        )

                                    # Define kwargs for the model plots
                                    upper_surface_kwargs = {
                                        "rstride": 1,
                                        "cstride": 1,
                                        "cmap": 'plasma',
                                        "linewidth": 0,
                                        "antialiased": True,
                                        "alpha": 1,  # Set transparency so we can see the overlap
                                        "zorder": 2
                                    }

                                    # Plot the first surface
                                    ax.plot_surface(
                                        x_upper,
                                        y_upper,
                                        z_upper,
                                            **upper_surface_kwargs
                                            )

                                # normal plotting without the plane
                                else:
                                    # # plot model
                                    ax.plot_surface(
                                        all_variables_dict[variables_compared_names[0]]["mesh"],
                                        all_variables_dict[variables_compared_names[1]]["mesh"],
                                        predicted_output,
                                        cmap="plasma"
                                        )


                                # plot ground truth

                                if plot_args["Surface"]["plot_ground_truth"]:

                                    # grab the values for slicing
                                    # import model_config 
                                    model_config_dict = json.load(open(model_path + "/model_config.json", "r"))

                                    # combine real x with real y
                                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                                    # slice data to get the data points where the fixed variable is the same.                    
                                    real_data_slice = full_real_data.copy()
                                    # iterate over the fixed variables slicing the data to the mid points
                                    for fixed_variable in variables_fixed.keys():

                                        real_data_slice = real_data_slice[
                                            real_data_slice[fixed_variable] == level
                                        ]


                                    # make sure x and y are the same as the model grid
                                    x_real_data = real_data_slice[variables_compared_names[0]]
                                    y_real_data = real_data_slice[variables_compared_names[1]]
                                    # extract the y data
                                    z_real_data = real_data_slice[self.Y_name]
                                    
                                    #plot
                                    ax.scatter(x_real_data, y_real_data, z_real_data)


                                ax.set_title(
                                    working_var+": "+str(level) + " " + design_parameters_dict["Variables"][working_var]["Units"],
                                    loc="center", fontsize=plot_args["Surface"]["Plot_title_font_size"])


                                fig.suptitle(
                                    "Surface Plot of Model Fit: "+ variables_compared_names[0]+" vs "+variables_compared_names[1], fontsize= plot_args["Surface"]["Suptitle_font_size"])

                                plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
                                plt.savefig(model_path + "surface_plots/"+working_var + "/surface_plot_"+experiment_description+"_"+working_var+"_"+str(i)+".png")
                                plt.clf()

                    ## now all plots have been made make gif
                    if plot_args["Surface"]["GIF"]:
                        import imageio

                        # Directory containing .png files
                        png_dir = model_path + "surface_plots/"+working_var
                        # Output GIF file name
                        gif_name = variables_compared_names[0]+"_vs_"+variables_compared_names[1]+".gif"

                        images = []
                        for file_name in sorted(os.listdir(png_dir)):
                            if file_name.endswith('.png'):
                                file_path = os.path.join(png_dir, file_name)
                                images.append(imageio.imread(file_path))


                        # Save the images as a GIF with the specified duration between frames
                        imageio.mimsave(os.path.join(png_dir, gif_name), images, 'GIF', fps=plot_args["Surface"]["fps"])                          
                                                                                    
        else:
            raise ValueError("Shape of X Data unknown.")


    def get_all_contour_plots(
        self,
        experiment_description,
        model_path,
        project_path,
        plot_args
        ):
            
        
        ## first create the directories for the individual contour plots
        # Extract the array and iterate through it to create dir
        for input_var in self.X_names:
            dir_path = os.path.join(model_path, "contour_plots", input_var)
            # Check if the directory exists
            if os.path.exists(dir_path):
                # If it does, delete it recursively
                shutil.rmtree(dir_path)
            # Now, create the directory
            os.makedirs(dir_path)

        # import design parameters
        design_parameters_dict = json.load(open(project_path + "design_parameters.json", 'r'))


        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        if len(self.X_names) == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif len(self.X_names) == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif len(self.X_names) >= 3:

            # import model config to get unique levels
            model_config_dict = json.load(open(model_path + "model_config.json", "r"))
            unique_levels_dict = model_config_dict["Unique_Levels"]

            variables = list(itertools.combinations(self.X_names, r=2))
            
            # set the z axis limit
            df_combinations = self.get_predicted_Y_for_all_level_combinations(model_path)

            z_axis_max = df_combinations[self.Y_name].max() * 1.2
            z_axis_min = df_combinations[self.Y_name].min() * 0.8

            ### mesh generation for each subplot.
            for combination in tqdm(variables, desc="Generating Plots"):

                                    
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared

                # get unique levels of combination
                # Filter unique_levels_dict to include only keys that are in variables_fixed
                variables_fixed_unique_levels_dict = {key: unique_levels_dict[key] for key in variables_fixed if key in unique_levels_dict}

                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data - real data not the feature matrices
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 100)
                        }
                    variables_compared[variable] = dict_for_assignment

                
                # get variable_meshes
                variable_arrays = [variable_dict["initial_array"] for variable_name, variable_dict in variables_compared.items()]
                variable_meshes = np.meshgrid(variable_arrays[0], variable_arrays[1])

                # assign the mesh to the appropriate variable
                for variable, mesh in zip(variables_compared, variable_meshes):

                    variables_compared[variable] = {
                        "mesh" : mesh,
                        "mesh_flat" : mesh.flatten()
                    }

                # get the variable component names
                variables_compared_names = list(combination)

                ##### this is where we iterate over the fixed variables levels to generate each subplot.
                # in order to create every possible slice, we fix one variable whilst we iterate over the over.
                # this one fixes the root variable



                for variable in variables_fixed:
                    # we then iterate again to get the "working var"
                    for working_var in variables_fixed:
                        # if they match, skip
                        if (variable == working_var) and (len(variables_fixed) != 1):
                            pass 
                        else:
                            
                            #iterate over the unique levels of the working var
                            for i, level in enumerate(variables_fixed_unique_levels_dict[working_var]):


                                # generate the prediction meshes
                                predicted_output, all_variables_dict = self.generate_x_y_z_prediction_meshes(
                                        fixed_level = level,
                                        fixed_variable_name = variable,
                                        fixed_variables_dict = variables_fixed,
                                        variables_compared_names = variables_compared_names,
                                        variables_compared_dict = variables_compared                                        )



                                # Create the surface plot
                                # Define your desired figsize (width, height in inches)
                                figsize = (10, 10)

                                # Create a figure with subplots
                                fig = plt.figure(figsize=(
                                    plot_args["Contour"]["Fig_Size_x"],
                                    plot_args["Contour"]["Fig_Size_y"]
                                    )
                                    )

                                # begin plot
                                ax = fig.add_subplot(111)

                                # axis limits
                                ax.set_xlim(
                                    xmin = min(variables_compared[variables_compared_names[0]]["mesh_flat"]),
                                    xmax = max(variables_compared[variables_compared_names[0]]["mesh_flat"]),
                                    )
                                ax.set_ylim(
                                    ymin = min(variables_compared[variables_compared_names[1]]["mesh_flat"]),
                                    ymax = max(variables_compared[variables_compared_names[1]]["mesh_flat"]),
                                    )


                                # axis labels
                                ax.set_xlabel(variables_compared_names[0], fontsize = plot_args["Contour"]["Axis_label_font_size"])
                                ax.set_ylabel(variables_compared_names[1], fontsize = plot_args["Contour"]["Axis_label_font_size"])
                                
                                # set ticks
                                if plot_args["Contour"]["Ticks_use_levels"]:
                                    ax.set_xticks(unique_levels_dict[variables_compared_names[0]])
                                    ax.tick_params(axis='x', labelsize=plot_args["Contour"]["Ticks_label_size"])

                                    ax.set_yticks(unique_levels_dict[variables_compared_names[1]])
                                    ax.tick_params(axis='y', labelsize=plot_args["Contour"]["Ticks_label_size"])


                                # Create the contour plot
                                ax.contourf(
                                        all_variables_dict[variables_compared_names[0]]["mesh"],
                                        all_variables_dict[variables_compared_names[1]]["mesh"],
                                        predicted_output,
                                        cmap="plasma",
                                        levels=20
                                        )

                                #plt.colorbar()

                                if plot_args["Contour"]["Grid"]:
                                    plt.grid(
                                        color=plot_args["Contour"]["Grid_colour"],
                                        linestyle=plot_args["Contour"]["Grid_linestyle"],
                                        linewidth=plot_args["Contour"]["Grid_linewidth"]
                                        )  # Customize the grid as needed


                                # plot ground truth

                                if plot_args["Contour"]["plot_ground_truth"]:

                                    # grab the values for slicing
                                    # import model_config 
                                    model_config_dict = json.load(open(model_path + "/model_config.json", "r"))

                                    # combine real x with real y
                                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                                    # slice data to get the data points where the fixed variable is the same.                    
                                    real_data_slice = full_real_data.copy()
                                    # iterate over the fixed variables slicing the data to the mid points
                                    for fixed_variable in variables_fixed.keys():

                                        real_data_slice = real_data_slice[
                                            real_data_slice[fixed_variable] == level
                                        ]


                                    # make sure x and y are the same as the model grid
                                    x_real_data = real_data_slice[variables_compared_names[0]]
                                    y_real_data = real_data_slice[variables_compared_names[1]]
                                    # extract the y data
                                    z_real_data = real_data_slice[self.Y_name]
                                    
                                    #plot
                                    ax.scatter(x_real_data, y_real_data, z_real_data)


                                ax.set_title(
                                    working_var+": "+str(level) + " " + design_parameters_dict["Variables"][working_var]["Units"],
                                    loc="center", fontsize=plot_args["Contour"]["Plot_title_font_size"])


                                fig.suptitle(
                                    "Contour Plot of Model Fit: "+ variables_compared_names[0]+" vs "+variables_compared_names[1], fontsize= plot_args["Contour"]["Suptitle_font_size"])

                                plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
                                plt.savefig(model_path + "contour_plots/"+working_var + "/contour_plot_"+experiment_description+"_"+working_var+"_"+str(i)+".png")
                                plt.close()

                    ## now all plots have been made make gif
                    if plot_args["Surface"]["GIF"]:
                        import imageio

                        # Directory containing .png files
                        png_dir = model_path + "contour_plots/"+working_var
                        # Output GIF file name
                        gif_name = variables_compared_names[0]+"_vs_"+variables_compared_names[1]+".gif"

                        images = []
                        for file_name in sorted(os.listdir(png_dir)):
                            if file_name.endswith('.png'):
                                file_path = os.path.join(png_dir, file_name)
                                images.append(imageio.imread(file_path))


                        # Save the images as a GIF with the specified duration between frames
                        imageio.mimsave(os.path.join(png_dir, gif_name), images, 'GIF', fps=plot_args["Contour"]["fps"])                          
                                                                                    
        else:
            raise ValueError("Shape of X Data unknown.")

    