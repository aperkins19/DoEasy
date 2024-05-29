import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import sympy as sp
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import itertools

class LinearModel:
    """
    LinearModel - A class for fitting and using linear regression models.

    This class encapsulates the process of fitting a linear regression model
    to provided training data and then using the fitted model to make predictions.

    Attributes:
        X_true (array-like): The feature matrix of shape (n_samples, n_features)
            containing the training input data.
        Y_true (array-like): The target vector of shape (n_samples,) containing
            the corresponding true target values.
        mse (float): Mean squared error of the fitted model's predictions on the
            training data.
        r2 (float): R-squared score of the fitted model on the training data.

    Methods:
        fit(): Fit the linear regression model to the training data and calculate
            MSE and R-squared scores.
        predict(New_X): Use the fitted model to make predictions on new data.

    Example:
        X_train = ...  # Training features
        Y_train = ...  # True target values
        X_new = ...    # New data for prediction

        model = LinearModel(X_train, Y_train)
        model.fit()
        predictions = model.predict(X_new)
    """
    def __init__(self, X, Y, project_path):
        """
        Initialize a LinearModel object.

        Args:
            X (array-like): Training feature matrix of shape (n_samples, n_features).
            Y (array-like): True target vector of shape (n_samples,).
        """
        self.X_true = X
        self.Y_true = Y
        self.model_object = LinearRegression()

        self.X_names = X.columns.values.tolist()
        self.Y_names = Y .columns.values.tolist()

        self.mse = None
        self.r2 = None
        
        # get centerpoint
        self.centerpoint = pd.read_csv(project_path + "Datasets/tidy_dataset.csv").loc[lambda df: df["DataPointType"] == "Center"].iloc[0]


    def fit(self):
        """
        Fit the linear regression model to the training data.

        This method uses the training data to fit the linear regression model
        and calculates the mean squared error (MSE) and R-squared (R2) scores.
        """

        self.model_object.fit(self.X_true, self.Y_true)

        self.y_predicted = self.model_object.predict(self.X_true)

        self.mse = mean_squared_error(self.Y_true, self.y_predicted)
        self.r2 = r2_score(self.Y_true, self.y_predicted)

    def predict(self, New_X):
        """
        Use the fitted model to make predictions on new data.

        Args:
            New_X (array-like): New feature matrix of shape (n_samples, n_features)
                for which predictions are to be made.

        Returns:
            array-like: Predicted target values for the new data.
        """
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")


        predicted_y = self.model_object.predict(New_X)

        return predicted_y


    def model_performance(self, model_id_string, project_path, feature_to_model, modeldirectory, get_plots = False):
        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")

        # Predict the outputs using the trained model
        predicted_output = self.model_object.predict(self.X_true)

        performance_df = pd.DataFrame()
        performance_df["Observations"] = self.Y_true
        performance_df["Predictions"] = predicted_output
        performance_df["Residuals"] = performance_df["Observations"] - performance_df["Predictions"]
        performance_df["Zeros"] = 0


        if get_plots:

            fig, ax = plt.subplots(nrows=1, ncols=2)
            fig.set_figheight(5)
            fig.set_figwidth(10)

            fig.suptitle("Model Performance", fontsize=18)

            # observations vs predictions
            ax[0].set_title("Observed vs Predicted")

            # begin plot
            sns.scatterplot(
                data = performance_df,
                x = "Predictions",
                y = "Observations",
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

            plt.savefig(modeldirectory + "/" +model_id_string+"_truth_vs_preds.png")

            plt.clf()

        return performance_df

    def find_max_y(self, candidate_array_size:int):
        """
        Find the Input Values that Maximize the Predicted Output.

        This method determines the combination of input values from the original feature space that results in the
        highest predicted output value according to the trained polynomial regression model.

        Returns:
            Series: A Pandas Series containing the input values that maximize the predicted output, along with the
            associated feature names and a "Predicted_Max" label.
        """
        dim_range_list = []
        # for every X dim, create an array from exp data min to exp data ma# add to list
        for name in self.X_names:
            dim = np.linspace(self.X_true[name].min(), self.X_true[name].max(), candidate_array_size)
            dim_range_list.append(dim)

        # unpack dim list to create a meshgrid for each dim.
        meshgrid = np.meshgrid(*dim_range_list)

        # Create input data
        # flattens each mesh and stacks to create 2D array where every sub array is one data point
        # [[X1,X2,X3]
        #  [X1,X2,X3]
        #  [X1,X2,X3]]
        input_data = np.column_stack([dim.flatten() for dim in meshgrid])

        # put in df to give feature names
        input_data = pd.DataFrame(input_data, columns = self.X_names)

        # Predict the outputs using the trained model
        # output
        #  [[Y1]
        #   [Y1]
        #   [Y1]]
        predicted_output = self.model_object.predict(input_data)

        # add the 1D array of predicted outputs as a new column to the input array
        #  [[X1,X2,X3,Y1]
        #   [X1,X2,X3,Y1]
        #   [X1,X2,X3,Y1]]
        predicted_dataset = np.concatenate((input_data, predicted_output), axis=1)

        # Extract Y1 values from the data array
        # predicted_dataset[:, -1]
        # Find the index of the row with the maximum Y1 value
        # np.argmax(predicted_dataset[:, -1])
        predicted_dataset_df = pd.DataFrame(predicted_dataset, columns=self.X_names+self.Y_names)
        return pd.Series(
            predicted_dataset[np.argmax(predicted_dataset[:, -1])],
            index= self.X_names + ["PredictedResponse"]
            ), predicted_dataset_df


    def get_surface_plots_as_figure(self, experiment_description, modelpath, feature_to_model, plot_ground_truth = False):

        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        
        if self.X_true.shape[1] == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif self.X_true.shape[1] == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif self.X_true.shape[1] >= 3:

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

            for combination, ax in zip(variables, axes.flat):
                
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared
                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}


                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 50)
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
                
                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(input_data)

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

                    # combine real x with real y
                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                    # slice data to get the data points where the fixed variable is the same.                    
                    real_data_slice = full_real_data.copy()
                    # iterate over the fixed variables slicing the data to the mid points
                    for fixed_variable in variables_fixed.keys():

                        ## legacy code 
                        #print(real_data_slice[
                        #    real_data_slice[fixed_variable] ==self.centerpoint[fixed_variable]])
                        #real_data_slice = real_data_slice[
                        #    real_data_slice[fixed_variable] == np.median(np.sort(self.X_true[fixed_variable].unique()))]
                        real_data_slice = real_data_slice[
                            real_data_slice[fixed_variable] == self.centerpoint[fixed_variable]]


                    # make sure x and y are the same as the model grid
                    x_real_data = real_data_slice[variables_compared_names[0]]
                    y_real_data = real_data_slice[variables_compared_names[1]]
                    # extract the y data
                    z_real_data = real_data_slice[self.Y_names[0]]
                    
                    #print(x_real_data)

                    #plot
                    ax.scatter(x_real_data, y_real_data, z_real_data)

                ax.view_init(azim = 30, elev = 5)

                # Create the contour plot
                ax.set_xlabel(variables_compared_names[0])
                ax.set_ylabel(variables_compared_names[1])
                ax.set_zlabel(self.Y_names[0])
                ax.set_title(
                    variables_compared_names[0]+" vs "+variables_compared_names[1],
                    loc="center")


            # Remove empty subplots if necessary
            if is_odd:
                fig.delaxes(axes.flat[-1])

            fig.suptitle("Surface Plots of Model Fit: " + experiment_description, fontsize= 15)

            plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
            plt.savefig(modelpath + "surface_plots/" +  "surface_plot_figure_"+experiment_description+".png")
            plt.clf()
                
        
        else:
            raise ValueError("Shape of X Data unknown.")


    def get_contour_plots_as_figure(self, experiment_description, modelpath, feature_to_model, plot_ground_truth = False):

        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        if self.X_true.shape[1] == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif self.X_true.shape[1] == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif self.X_true.shape[1] >= 3:

            variables = list(itertools.combinations(self.X_names, r=2))

            # as the plots are build, save them for figure construction after the loop.
            number_of_combinations = len(variables)

            # Define your desired figsize (width, height in inches)
            figsize = (10, 2 * number_of_combinations)

            is_odd = (number_of_combinations % 2) != 0
            if not is_odd:
                # Create a figure with subplots
                fig, axes = plt.subplots(int(number_of_combinations/2), 2, figsize=figsize)  # 2x2 grid for subplots

            # Create a figure with subplots
            fig, axes = plt.subplots(round(number_of_combinations/2), 2, figsize=figsize)  # 2x2 grid for subplots

            for combination, ax in zip(variables, axes.flat):
                
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared
                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 50)
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

                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(input_data)

                # get the variable component names
                variables_compared_names = list(combination)

                # Reshape the predicted outputs for contour plotting
                predicted_output = predicted_output.reshape(
                    all_variables_dict[variables_compared_names[0]]["mesh"].shape
                    )

                # begin plot
                ax = fig.add_subplot(ax)

                # plot model
                ax.contourf(
                    all_variables_dict[variables_compared_names[0]]["mesh"],
                    all_variables_dict[variables_compared_names[1]]["mesh"],
                    predicted_output,
                    levels=20,
                    cmap='viridis'
                    )

                # plot ground truth

                if plot_ground_truth:

                    # combine real x with real y
                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                    # slice data to get the data points where the fixed variable is the same.                    
                    real_data_slice = full_real_data.copy()
                    # iterate over the fixed variables slicing the data to the mid points
                    for fixed_variable in variables_fixed.keys():
                        real_data_slice = real_data_slice[
                            real_data_slice[fixed_variable] == np.median(np.sort(self.X_true[fixed_variable].unique()))]

                    # make sure x and y are the same as the model grid
                    x_real_data = real_data_slice[variables_compared_names[0]]
                    y_real_data = real_data_slice[variables_compared_names[1]]
                    # extract the y data
                    z_real_data = real_data_slice[self.Y_names[0]]

                    #plot
                    ax.scatter(x_real_data, y_real_data, z_real_data)

                # Create the contour plot
                ax.set_xlabel(variables_compared_names[0])
                ax.set_ylabel(variables_compared_names[1])
                #ax.set_zlabel(self.Y_names[0])
                ax.set_title(
                    variables_compared_names[0]+" vs "+variables_compared_names[1],
                    loc="center")

                #plt.title(f"Surface Plot for "+variables_compared_names[0]+" vs "+variables_compared_names[1])
                #plt.savefig("/app/analysis_output/surface_plots/"+variables_compared_names[0]+" vs "+variables_compared_names[1]+"_surface_plot.png")

                # save the plot
                #plt.clf()



            # Remove empty subplots if necessary
            if is_odd:
                fig.delaxes(axes.flat[-1])

            fig.suptitle("Contour Plots of Model Fit: " + experiment_description, fontsize= 15)

            plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
            plt.savefig(modelpath + "contour_plots/" +  "contour_plot_figure_"+experiment_description+".png")
            plt.clf()
                
        
        else:
            raise ValueError("Shape of X Data unknown.")




class PolyModel:
    """
    Polynomial Regression Model with Feature Transformation

    This class encapsulates a polynomial regression model with a flexible degree of polynomial features. It performs
    feature transformation, model fitting, and provides methods to evaluate the model's performance.


    Methods:
        fit_model(): Fit the polynomial regression model.

    """
    def __init__(self, X, Y, degree, project_path):
        """
        Initialize the Polynomial Regression Model.

        Args:
            X (DataFrame): Input feature DataFrame.
            Y (Series or DataFrame): Target variable(s).
            degree (int): Degree of polynomial features to be used.

        Attributes:
            X_true (DataFrame): The input features DataFrame.
            Y_true (Series or DataFrame): The target variable(s).
            degree (int): Degree of polynomial features.
            PolyFeatureTransformer (PolynomialFeatures): The polynomial feature transformer object.
            X_true_poly_features (array-like): Transformed polynomial features of X_true.
            X_names (list): List of original feature column names.
            feature_names (list): List of feature names after transformation.
            Y_names (list): List of target variable column names.
            model_object (LinearRegression): Linear regression model object.
            mse (float): Mean squared error of the model's predictions.
            r2 (float): R-squared coefficient indicating the model's goodness of fit.
        """

        self.degree = degree
        self.PolyFeatureTransformer = PolynomialFeatures(
            degree = self.degree,
            #include_bias = False,
            #interaction_only = True
            )

        self.X_true = X
        self.X_names = X.columns.values.tolist()

        self.X_true_poly_features = self.PolyFeatureTransformer.fit_transform(self.X_true)
        # extracts the column names of the X df and stores as list
        self.feature_names = self.PolyFeatureTransformer.get_feature_names_out(input_features = self.X_names)

        self.Y_true = Y
        self.Y_names = Y.columns.values.tolist()
        
        self.model_object = LinearRegression()

        self.mse = None
        self.r2 = None

        # get centerpoint
        self.centerpoint = pd.read_csv(project_path + "Datasets/tidy_dataset.csv").loc[lambda df: df["DataPointType"] == "Center"].iloc[0]

    def fit(self):

        """
        Fit the polynomical lballs = itertools.combinations(['red', 'green', 'blue', 'yellow'], r=3)inear regression model to the training data.

        Uses the polynomial X features generated on __init__
        and calculates the mean squared error (MSE) and R-squared (R2) scores.
        """

        self.model_object.fit(self.X_true_poly_features, y = self.Y_true)

        self.y_predicted = self.model_object.predict(self.X_true_poly_features)

        self.mse = mean_squared_error(self.Y_true, self.y_predicted)
        self.r2 = r2_score(self.Y_true, self.y_predicted)
    
    def predict(self, New_X):

        """
        Use the fitted model to make predictions on new data.

        Args:
            New_X (array-like): New feature matrix of shape (n_samples, n_features)
                for which predictions are to be made.

        Returns:
            array-like: Predicted target values for the new data.
        """

        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")

        predicted_y = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(New_X))

        return predicted_y

    def find_max_y(self, candidate_array_size:int):
        """
        Find the Input Values that Maximize the Predicted Output.

        This method determines the combination of input values from the original feature space that results in the
        highest predicted output value according to the trained polynomial regression model.

        Returns:
            Series: A Pandas Series containing the input values that maximize the predicted output, along with the
            associated feature names and a "Predicted_Max" label.
        """
        dim_range_list = []
        # for every X dim, create an array from exp data min to exp data ma# add to list
        for name in self.X_names:
            dim = np.linspace(self.X_true[name].min(), self.X_true[name].max(), candidate_array_size)
            dim_range_list.append(dim)

        # unpack dim list to create a meshgrid for each dim.
        meshgrid = np.meshgrid(*dim_range_list)

        # Create input data
        # flattens each mesh and stacks to create 2D array where every sub array is one data point
        # [[X1,X2,X3]
        #  [X1,X2,X3]
        #  [X1,X2,X3]]
        input_data = np.column_stack([dim.flatten() for dim in meshgrid])

        # Predict the outputs using the trained model
        # output
        #  [[Y1]
        #   [Y1]
        #   [Y1]]
        predicted_output = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(input_data))

        # add the 1D array of predicted outputs as a new column to the input array
        #  [[X1,X2,X3,Y1]
        #   [X1,X2,X3,Y1]
        #   [X1,X2,X3,Y1]]
        predicted_dataset = np.concatenate((input_data, predicted_output), axis=1)

        # Extract Y1 values from the data array
        # predicted_dataset[:, -1]
        # Find the index of the row with the maximum Y1 value
        # np.argmax(predicted_dataset[:, -1])
        predicted_dataset_df = pd.DataFrame(predicted_dataset, columns=self.X_names+self.Y_names)
        return pd.Series(
            predicted_dataset[np.argmax(predicted_dataset[:, -1])],
            index= self.X_names + ["PredictedResponse"]
            ), predicted_dataset_df


    def get_contour_interaction_plots(
        self,
        experiment_description,
        modelpath,
        feature_to_model,
        plot_ground_truth):


        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        
        if self.X_true.shape[1] == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif self.X_true.shape[1] == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif self.X_true.shape[1] >= 3:
            
            plot_list = []
            for name in self.X_names:

                # leave one out method - only works for 3x variables
                #stores the names of the x and y variables in x_y and the name of the variable to be fixed in z.
                x_y = self.X_names.copy()
                x_y.remove(name)
                z = name
                # Choose the value of the third variable to keep constant
                # fixes the value at the mean of the data range
                Fixed_Z_Value = self.X_true[z].mean()


                # Generate a grid of values for the chosen variables
                x_var = np.linspace(self.X_true[x_y[0]].min(), self.X_true[x_y[0]].max(), 50)
                y_var = np.linspace(self.X_true[x_y[1]].min(), self.X_true[x_y[1]].max(), 50)
                x_grid, y_grid = np.meshgrid(x_var, y_var)

                # produces an array o_center
                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(input_data))
                #predicted_output = self.model_object.predict(input_data)

                # Reshape the predicted outputs for contour plotting
                predicted_output = predicted_output.reshape(x_grid.shape)

                # Create the contour plot
                plt.contourf(x_grid, y_grid, predicted_output, levels=20, cmap='viridis')
                plt.colorbar()
                plt.xlabel(x_y[0])
                plt.ylabel(x_y[1])
                plt.title(f"Contour Plot for "+x_y[0]+" vs "+x_y[1]+". "+z+": "+str(round(Fixed_Z_Value,2)))
                plt.savefig(modelpath + "/contour_plots/" + x_y[0]+" vs "+x_y[1]+"_contour_plot.png")
                plt.clf()

        else:
            raise ValueError("Shape of X Data unknown.")


    def get_contour_plots_as_figure(self, experiment_description, modelpath, feature_to_model, plot_ground_truth = False):

        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        if self.X_true.shape[1] == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif self.X_true.shape[1] == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif self.X_true.shape[1] >= 3:

            variables = list(itertools.combinations(self.X_names, r=2))

            # as the plots are build, save them for figure construction after the loop.
            number_of_combinations = len(variables)

            # Define your desired figsize (width, height in inches)
            figsize = (10, 2 * number_of_combinations)

            is_odd = (number_of_combinations % 2) != 0
            if not is_odd:
                # Create a figure with subplots
                fig, axes = plt.subplots(int(number_of_combinations/2), 2, figsize=figsize)  # 2x2 grid for subplots

            # Create a figure with subplots
            fig, axes = plt.subplots(round(number_of_combinations/2), 2, figsize=figsize)  # 2x2 grid for subplots

            for combination, ax in zip(variables, axes.flat):
                
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared
                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 50)
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
                
                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(input_data))

                # get the variable component names
                variables_compared_names = list(combination)

                # Reshape the predicted outputs for contour plotting
                predicted_output = predicted_output.reshape(
                    all_variables_dict[variables_compared_names[0]]["mesh"].shape
                    )

                # begin plot
                ax = fig.add_subplot(ax)

                # plot model
                ax.contourf(
                    all_variables_dict[variables_compared_names[0]]["mesh"],
                    all_variables_dict[variables_compared_names[1]]["mesh"],
                    predicted_output,
                    levels=20,
                    cmap='viridis'
                    )

                # plot ground truth

                if plot_ground_truth:

                    # combine real x with real y
                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                    # slice data to get the data points where the fixed variable is the same.                    
                    real_data_slice = full_real_data.copy()
                    # iterate over the fixed variables slicing the data to the mid points
                    for fixed_variable in variables_fixed.keys():
                        real_data_slice = real_data_slice[
                            real_data_slice[fixed_variable] == np.median(np.sort(self.X_true[fixed_variable].unique()))]

                    # make sure x and y are the same as the model grid
                    x_real_data = real_data_slice[variables_compared_names[0]]
                    y_real_data = real_data_slice[variables_compared_names[1]]
                    # extract the y data
                    z_real_data = real_data_slice[self.Y_names[0]]

                    #plot
                    ax.scatter(x_real_data, y_real_data, z_real_data)

                # Create the contour plot
                ax.set_xlabel(variables_compared_names[0])
                ax.set_ylabel(variables_compared_names[1])
                #ax.set_zlabel(self.Y_names[0])
                ax.set_title(
                    variables_compared_names[0]+" vs "+variables_compared_names[1],
                    loc="center")

                #plt.title(f"Surface Plot for "+variables_compared_names[0]+" vs "+variables_compared_names[1])
                #plt.savefig("/app/analysis_output/surface_plots/"+variables_compared_names[0]+" vs "+variables_compared_names[1]+"_surface_plot.png")

                # save the plot
                #plt.clf()



            # Remove empty subplots if necessary
            if is_odd:
                fig.delaxes(axes.flat[-1])

            fig.suptitle("Contour Plots of Model Fit: " + experiment_description, fontsize= 15)

            plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
            plt.savefig(modelpath + "contour_plots/" +  "contour_plot_figure_"+experiment_description+".png")
            plt.clf()
                
        
        else:
            raise ValueError("Shape of X Data unknown.")

    def get_surface_plots_as_figure(self, experiment_description, modelpath, feature_to_model, plot_ground_truth = False):

        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        
        if self.X_true.shape[1] == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif self.X_true.shape[1] == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif self.X_true.shape[1] >= 3:

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

            for combination, ax in zip(variables, axes.flat):
                
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared
                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 50)
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
                
                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(input_data))

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

                    # combine real x with real y
                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                    # slice data to get the data points where the fixed variable is the same.                    
                    real_data_slice = full_real_data.copy()
                    # iterate over the fixed variables slicing the data to the mid points
                    for fixed_variable in variables_fixed.keys():

                        ## legacy code 
                        #print(real_data_slice[
                        #    real_data_slice[fixed_variable] ==self.centerpoint[fixed_variable]])
                        #real_data_slice = real_data_slice[
                        #    real_data_slice[fixed_variable] == np.median(np.sort(self.X_true[fixed_variable].unique()))]
                        real_data_slice = real_data_slice[
                            real_data_slice[fixed_variable] == self.centerpoint[fixed_variable]]


                    # make sure x and y are the same as the model grid
                    x_real_data = real_data_slice[variables_compared_names[0]]
                    y_real_data = real_data_slice[variables_compared_names[1]]
                    # extract the y data
                    z_real_data = real_data_slice[self.Y_names[0]]
                    
                    #print(x_real_data)

                    #plot
                    ax.scatter(x_real_data, y_real_data, z_real_data)

                ax.view_init(azim = 30, elev = 5)

                # Create the contour plot
                ax.set_xlabel(variables_compared_names[0])
                ax.set_ylabel(variables_compared_names[1])
                ax.set_zlabel(self.Y_names[0])
                ax.set_title(
                    variables_compared_names[0]+" vs "+variables_compared_names[1],
                    loc="center")


            # Remove empty subplots if necessary
            if is_odd:
                fig.delaxes(axes.flat[-1])

            fig.suptitle("Surface Plots of Model Fit: " + experiment_description, fontsize= 15)

            plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # Adjust the rect to prevent overlap with the title and legend
            plt.savefig(modelpath + "surface_plots/" +  "surface_plot_figure_"+experiment_description+".png")
            plt.clf()
                
        
        else:
            raise ValueError("Shape of X Data unknown.")

    def get_surface_plots(self, experiment_description, modelpath, feature_to_model, plot_ground_truth = False):

        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")        

        # determine the amount of X variables

        
        if self.X_true.shape[1] == 1:
            raise ValueError("Only one X variable, contour plot not applicable.")

        elif self.X_true.shape[1] == 2:
            raise ValueError("Single contour plot to be implemented.")

        elif self.X_true.shape[1] == 3:

            variables = list(itertools.combinations(self.X_names, r=2))


            for combination in variables:
                
                # convert to sets to extract variable compared and fixed
                variables_compared = set(combination)
                variables_fixed = set(self.X_names) - variables_compared
                
                # convert to dicts for population
                variables_compared = {element: None for element in variables_compared}
                variables_fixed = {element: None for element in variables_fixed}

                # assign an array to the variables compared for the purpose of creating the meshes
                for variable in variables_compared:
                    # assign an array of min and max from variable data
                    dict_for_assignment = {
                    "initial_array":
                        np.linspace(self.X_true[variable].min(), self.X_true[variable].max(), 50)
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
                
                # Predict the outputs using the trained model
                predicted_output = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(input_data))

                # get the variable component names
                variables_compared_names = list(combination)

                # Reshape the predicted outputs for contour plotting
                predicted_output = predicted_output.reshape(
                    all_variables_dict[variables_compared_names[0]]["mesh"].shape
                    )

                # begin plot
                ax = plt.axes(projection="3d")

                # plot model
                ax.plot_surface(
                    all_variables_dict[variables_compared_names[0]]["mesh"],
                    all_variables_dict[variables_compared_names[1]]["mesh"],
                    predicted_output,
                    cmap="plasma"
                    )

                # plot ground truth

                if plot_ground_truth:

                    # combine real x with real y
                    full_real_data = pd.concat([self.X_true, self.Y_true], axis=1)

                    # slice data to get the data points where the fixed variable is the same.                    
                    real_data_slice = full_real_data.copy()
                    # iterate over the fixed variables slicing the data to the mid points
                    for fixed_variable in variables_fixed.keys():
                        real_data_slice = real_data_slice[
                            real_data_slice[fixed_variable] == np.median(np.sort(self.X_true[fixed_variable].unique()))]

                    # make sure x and y are the same as the model grid
                    x_real_data = real_data_slice[variables_compared_names[0]]
                    y_real_data = real_data_slice[variables_compared_names[1]]
                    # extract the y data
                    z_real_data = real_data_slice[self.Y_names[0]]

                    #plot
                    ax.scatter(x_real_data, y_real_data, z_real_data)

                ax.view_init(azim = 20, elev = 5)

                # Create the contour plot
                ax.set_xlabel(variables_compared_names[0])
                ax.set_ylabel(variables_compared_names[1])
                ax.set_zlabel(self.Y_names[0])
                ax.set_title(
                    variables_compared_names[0]+" vs "+variables_compared_names[1],
                    loc="center"
                    )

                plt.title(f"Surface Plot for "+variables_compared_names[0]+" vs "+variables_compared_names[1])
                plt.savefig(modelpath + "/surface_plots/"  + variables_compared_names[0]+" vs "+variables_compared_names[1]+"_surface_plot_"+experiment_description+".png")
                
                # save the plot
                plt.clf()
        else:
            raise ValueError("Shape of X Data unknown.")
            

    def model_performance(self, model_id_string, project_path, feature_to_model, modeldirectory, get_plots = False):
        # first check if model is fitted.
        if self.r2 is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first. ")

        # Predict the outputs using the trained model
        input_array = self.X_true.to_numpy()
        predicted_output = self.model_object.predict(self.PolyFeatureTransformer.fit_transform(input_array))

        performance_df = pd.DataFrame()
        performance_df["Observations"] = self.Y_true
        performance_df["Predictions"] = predicted_output
        performance_df["Residuals"] = performance_df["Observations"] - performance_df["Predictions"]
        performance_df["Zeros"] = 0


        if get_plots:

            fig, ax = plt.subplots(nrows=1, ncols=2)
            fig.set_figheight(5)
            fig.set_figwidth(10)

            fig.suptitle("Model Performance", fontsize=18)

            # observations vs predictions
            ax[0].set_title("Observed vs Predicted")

            # begin plot
            sns.scatterplot(
                data = performance_df,
                x = "Predictions",
                y = "Observations",
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

            plt.savefig(modeldirectory + "/" +model_id_string+"_truth_vs_preds.png")

            plt.clf()

        return performance_df
