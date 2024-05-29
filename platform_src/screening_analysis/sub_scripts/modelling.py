import pandas as pd
import numpy as np

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

        if '.' in feature:  # Interaction terms
            factors = feature.split('.')
            # Calculate the product of the factors
            interaction_term = feature_matrix[factors[0]]
            for factor in factors[1:]:
                interaction_term *= feature_matrix[factor]
            feature_matrix[f'{feature}'] = interaction_term

        elif '**' in feature:  # Polynomial terms
            factor, power = feature.split('**')
            feature_matrix[f'{feature}'] = feature_matrix[factor] ** int(power)

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

    print(model_features)
    print()

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
