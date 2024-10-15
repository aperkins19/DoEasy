import pandas as pd
import numpy as np
import sympy as sy

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error



class NaturalLogFitter:
    """
    A class for fitting data to a Natural Log function and performing related operations.

    This class provides methods for fitting data to a Natural Log function, predicting values using
    the fitted parameters, and calculating derivatives and specific points of interest from the function.

    Attributes:
    params (tuple or None): Fitted parameters obtained after performing curve fitting.
                           None if fitting hasn't been performed yet.
    """

    def __init__(self):
        """
        Initialize a NaturalLogFitter object.

        Initializes the instance with no fitted parameters.
        """

        self.params = None

    def natural_log_function(self, x, a, b):
        """
        Calculate the value of a Natural Log function.

        Parameters:
        t (array-like): Array of t values at which to calculate the function.
        a, b, c (float): Parameters of the natural_log_function function.

        Returns:
        ndarray: Array of function values corresponding to the input t values.
        """
        return a * np.log(x) + b


    def fit(self, x, y):
        """
        Fit data to the Natural Log function.

        Parameters:
        t (array-like): Array of t values representing the independent variable.
        y (array-like): Array of y values representing the dependent variable.

        Side Effects:
        - Updates the 'params' attribute with the fitted parameters.
        """
        self.params, _ = curve_fit(self.natural_log_function, x, y)

        self.mse = mean_squared_error(y, self.predict(x))


    def get_parameters(self):
        """
        Get the fitted parameters of the Natural Log function.

        Returns the fitted parameters obtained after performing the curve fitting process.

        Returns:
        tuple: Fitted parameters (a, b) of the Natural Log function.

        Raises:
        ValueError: If fitting has not been performed yet, raise an error.
        """
        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        return self.params

    def predict(self, x):
        """
        Predict y values using the fitted Natural Log function.

        Calculates predicted y values using the fitted parameters and the Natural Logd function
        for the given t values.

        Parameters:
        t (array-like): Array of t values for which to predict y values.

        Returns:
        ndarray: Array of predicted y values corresponding to the input t values.

        Raises:
        ValueError: If fitting has not been performed yet, raise an error.
        """
        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")

        a_fit, b_fit = self.params

        return self.natural_log_function(x, a_fit, b_fit)


    def get_fit_metrics(self, x, y):

        self.mse = mean_squared_error(y, self.predict(x))


    def get_max_yield(self, x):

        """
        Returns the maximum yield value predicted by the fitted Natural Log function.

        Calculates the predicted yield values using the fitted parameters and the Natural Log function,
        and returns the maximum yield value and its corresponding x value from the input array `x`.

        Parameters:
        x (array-like): Array of x values for which to calculate the predicted yield.

        Returns:
        float: X value corresponding to the maximum predicted yield.
        float: Maximum predicted yield value.
        """

        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")

        a_fit, b_fit = self.params
        y = self.natural_log_function(x, a_fit, b_fit)

        return x[-1], y[-1]



class GompertzFitter:
    """
    A class for fitting data to a Gompertz function and performing related operations.

    This class provides methods for fitting data to a Gompertz function, predicting values using
    the fitted parameters, and calculating derivatives and specific points of interest from the function.

    Attributes:
    params (tuple or None): Fitted parameters obtained after performing curve fitting.
                           None if fitting hasn't been performed yet.
    """

    def __init__(self):
        """
        Initialize a GompertzFitter object.

        Initializes the instance with no fitted parameters.
        """

        self.params = None

    def gompertz_function(self, x, a, b, c):
        """
        Calculate the value of a Gompertz function.

        Parameters:
        t (array-like): Array of t values at which to calculate the function.
        a, b, c (float): Parameters of the Gompertz function.

        Returns:
        ndarray: Array of function values corresponding to the input t values.
        """
        return a * np.exp(-b * np.exp(-c * x))

    def fit(self, x, y):
        """
        Fit data to the Gompertz function.

        Parameters:
        t (array-like): Array of t values representing the independent variable.
        y (array-like): Array of y values representing the dependent variable.

        Side Effects:
        - Updates the 'params' attribute with the fitted parameters.
        """
        self.params, _ = curve_fit(self.gompertz_function, x, y)

    def get_parameters(self):
        """
        Get the fitted parameters of the Gompertz function.

        Returns the fitted parameters obtained after performing the curve fitting process.

        Returns:
        tuple: Fitted parameters (a, b, c) of the Gompertz function.

        Raises:
        ValueError: If fitting has not been performed yet, raise an error.
        """
        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        return self.params

    def predict(self, x):
        """
        Predict y values using the fitted Gompertz function.

        Calculates predicted y values using the fitted parameters and the Gompertz function
        for the given t values.

        Parameters:
        t (array-like): Array of t values for which to predict y values.

        Returns:
        ndarray: Array of predicted y values corresponding to the input t values.

        Raises:
        ValueError: If fitting has not been performed yet, raise an error.
        """
        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")

        a_fit, b_fit, c_fit = self.params

        return self.gompertz_function(x, a_fit, b_fit, c_fit)

    def get_fit_metrics(self, x, y):

        self.mse = mean_squared_error(y, self.predict(x))

    def get_max_yield(self, x):

        """
        Returns the maximum yield value predicted by the fitted asymmetric sigmoid function.
        
        Calculates the predicted yield values using the fitted parameters and the asymmetric sigmoid function,
        and returns the maximum yield value and its corresponding x value from the input array `x`.
        
        Parameters:
        x (array-like): Array of x values for which to calculate the predicted yield.
        
        Returns:
        float: X value corresponding to the maximum predicted yield.
        float: Maximum predicted yield value.
        """

        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        
        a_fit, b_fit, c_fit = self.params
        y = self.gompertz_function(x, a_fit, b_fit, c_fit)
        
        return x[-1], y[-1]
    

    def get_max_rate(self, data_time_values):

        """
        Returns the maximum rate of the function.
        
        Calculates the first derivative of the function with respect to x, generates a linspace of x values
        using the time series of the data, generates corresponding y values, and returns the maximum rate
        and its corresponding time x value.
        
        Parameters:
        data_time_values (array-like): Time values from the data.
        
        Returns:
        float: Time x value at which the maximum rate occurs.
        float: Maximum rate value.
        """


        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        

        # declare symbolic parameters and function
        x, a, b, c = sy.symbols("x a b c", real = True)

        # re define function so that the exp can be differentiated
        f = a * sy.exp(-b * sy.exp(-c * x))


        # Calculate the symbolic derivative of the function with respect to x
        dfdx = sy.diff(f, x)

        # Convert the symbolic derivative to a callable function
        dfdx_f = sy.lambdify((x, a, b, c), dfdx)

        # Generate a range of x values for evaluation
        diff_x_values = np.linspace(1, max(data_time_values.to_numpy()), 100)
        
        # Evaluate the derivative function for the given parameters and x values
        y_values_dfdx_f = dfdx_f(diff_x_values, a=self.params[0], b=self.params[1], c=self.params[2])
        
        # Find the x value where the derivative function has its maximum value
        x_of_max_rate_of_reaction = diff_x_values[np.where(y_values_dfdx_f == max(y_values_dfdx_f))[0][0]]

        return x_of_max_rate_of_reaction, max(y_values_dfdx_f)


    def get_inflection_point(self, data_time_values):
        """
        Returns the 2nd inflection point of the function.
        
        Calculates the second derivative of the function with respect to x, generates a linspace of x values
        using the time series of the data, generates corresponding y values, and returns the minimum value
        and its corresponding time x value.
        
        Parameters:
        data_time_values (array-like): Time values from the data.
        
        Returns:
        float: Time x value at which the minimum occurs.
        float: Minimum value.
        """

        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        
        # declare symbolic parameters and function
        x, a, b, c = sy.symbols("x a b c", real = True)

        # re define function so that the exp can be differentiated
        f = a * sy.exp(-b * sy.exp(-c * x))

        # Calculate the symbolic 2nd derivative of the function with respect to x
        dfdx2 = sy.diff(f, x, 2)

        # Convert the symbolic 2nd derivative to a callable function
        dfdx2_f = sy.lambdify((x, a, b, c), dfdx2)

        # Generate a range of x values for evaluation
        diff_x_values = np.linspace(1, max(data_time_values.to_numpy()), 100)
        
        # Evaluate the 2nd derivative function for the given parameters and x values
        y_values_dfdx2_f = dfdx2_f(diff_x_values, a=self.params[0], b=self.params[1], c=self.params[2])
        
        # Find the x value where the 2nd derivative function has its minimum value
        x_of_min_2nd_derivative = diff_x_values[np.where(y_values_dfdx2_f == min(y_values_dfdx2_f))[0][0]]

        return x_of_min_2nd_derivative, min(y_values_dfdx2_f)

class AsymmetricSigmoidFitter:

    """
    A class for fitting data to an asymmetric sigmoid function and performing related operations.
    
    This class provides methods for fitting data to an asymmetric sigmoid function, predicting values using
    the fitted parameters, and calculating derivatives and specific points of interest from the function.
    
    Attributes:
    params (tuple or None): Fitted parameters obtained after performing curve fitting.
                           None if fitting hasn't been performed yet.
    """


    def __init__(self):
        """
        Initialize an AsymmetricSigmoidFitter object.
        
        Initializes the instance with no fitted parameters or coefficient
        """

        self.params = None
        self.coeff = None
    
    def asymmetric_sigmoid(self, x, a, b, c, d):
        """
        Calculate the value of an asymmetric sigmoid function.
        
        Parameters:
        x (array-like): Array of x values at which to calculate the function.
        a, b, c, d (float): Parameters of the sigmoid function.
        
        Returns:
        ndarray: Array of function values corresponding to the input x values.
        """
        return a + ((d - a) / ((1 + (x / c)**b)))
    
    def fit(self, x, y):

        """
        Fit data to the asymmetric sigmoid function.
        
        Parameters:
        x (array-like): Array of x values representing the independent variable.
        y (array-like): Array of y values representing the dependent variable.
        
        Side Effects:
        - Updates the 'params' attribute with the fitted parameters.
        - Updates the 'coefficient' attribute with the coefficient
        """
        self.params, self.coeff = curve_fit(self.asymmetric_sigmoid, x, y)
    
    def get_parameters(self):
        """
        Get the fitted parameters of the asymmetric sigmoid function.
        
        Returns the fitted parameters obtained after performing the curve fitting process.
        
        Returns:
        tuple: Fitted parameters (a, b, c, d) of the asymmetric sigmoid function.
        
        Raises:
        ValueError: If fitting has not been performed yet, raise an error.
        """
        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        return self.params

    def predict(self, x):
        """
        Predict y values using the fitted asymmetric sigmoid function.
        
        Calculates predicted y values using the fitted parameters and the asymmetric sigmoid function
        for the given x values.
        
        Parameters:
        x (array-like): Array of x values for which to predict y values.
        
        Returns:
        ndarray: Array of predicted y values corresponding to the input x values.
        
        Raises:
        ValueError: If fitting has not been performed yet, raise an error.
        """
        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        
        a_fit, b_fit, c_fit, d_fit = self.params
        
        return self.asymmetric_sigmoid(x, a_fit, b_fit, c_fit, d_fit)


    def get_fit_metrics(self, x, y):

        self.mse = mean_squared_error(y, self.predict(x))


    def get_max_yield(self, x):

        """
        Returns the maximum yield value predicted by the fitted asymmetric sigmoid function.
        
        Calculates the predicted yield values using the fitted parameters and the asymmetric sigmoid function,
        and returns the maximum yield value and its corresponding x value from the input array `x`.
        
        Parameters:
        x (array-like): Array of x values for which to calculate the predicted yield.
        
        Returns:
        float: X value corresponding to the maximum predicted yield.
        float: Maximum predicted yield value.
        """

        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        
        a_fit, b_fit, c_fit, d_fit = self.params

        y = self.asymmetric_sigmoid(x, a_fit, b_fit, c_fit, d_fit)
        
        return x[-1], y[-1]
    
    def get_max_rate(self, data_time_values):

        """
        Returns the maximum rate of the function.
        
        Calculates the first derivative of the function with respect to x, generates a linspace of x values
        using the time series of the data, generates corresponding y values, and returns the maximum rate
        and its corresponding time x value.
        
        Parameters:
        data_time_values (array-like): Time values from the data.
        
        Returns:
        float: Time x value at which the maximum rate occurs.
        float: Maximum rate value.
        """


        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        

        # declare symbolic parameters and function
        x, a, b, c, d = sy.symbols("x a b c d", real = True)
        f = self.asymmetric_sigmoid(x,a,b,c,d)


        # Calculate the symbolic derivative of the function with respect to x
        dfdx = sy.diff(f, x)

        # Convert the symbolic derivative to a callable function
        dfdx_f = sy.lambdify((x, a, b, c, d), dfdx)

        # Generate a range of x values for evaluation
        diff_x_values = np.linspace(1, max(data_time_values.to_numpy()), 100)
        
        # Evaluate the derivative function for the given parameters and x values
        y_values_dfdx_f = dfdx_f(diff_x_values, a=self.params[0], b=self.params[1], c=self.params[2], d=self.params[3])
        
        # Find the x value where the derivative function has its maximum value
        x_of_max_rate_of_reaction = diff_x_values[np.where(y_values_dfdx_f == max(y_values_dfdx_f))[0][0]]

        return x_of_max_rate_of_reaction, max(y_values_dfdx_f)



    def get_inflection_point(self, data_time_values):
        """
        Returns the 2nd inflection point of the function.
        
        Calculates the second derivative of the function with respect to x, generates a linspace of x values
        using the time series of the data, generates corresponding y values, and returns the minimum value
        and its corresponding time x value.
        
        Parameters:
        data_time_values (array-like): Time values from the data.
        
        Returns:
        float: Time x value at which the minimum occurs.
        float: Minimum value.
        """

        if self.params is None:
            raise ValueError("Fit not performed yet. Call the 'fit' method first.")
        
        # declare symbolic parameters and function
        x, a, b, c, d = sy.symbols(" x a b c d", real = True)
        f = self.asymmetric_sigmoid(x,a,b,c,d)

        # Calculate the symbolic 2nd derivative of the function with respect to x
        dfdx2 = sy.diff(f, x, 2)

        # Convert the symbolic 2nd derivative to a callable function
        dfdx2_f = sy.lambdify((x, a, b, c, d), dfdx2)

        # Generate a range of x values for evaluation
        diff_x_values = np.linspace(1, max(data_time_values.to_numpy()), 100)
        
        # Evaluate the 2nd derivative function for the given parameters and x values
        y_values_dfdx2_f = dfdx2_f(diff_x_values, a=self.params[0], b=self.params[1], c=self.params[2], d=self.params[3])
        
        # Find the x value where the 2nd derivative function has its minimum value
        x_of_min_2nd_derivative = diff_x_values[np.where(y_values_dfdx2_f == min(y_values_dfdx2_f))[0][0]]

        return x_of_min_2nd_derivative, min(y_values_dfdx2_f)
