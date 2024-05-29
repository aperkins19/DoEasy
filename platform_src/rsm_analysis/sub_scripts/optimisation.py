import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def simple_greedy_search_multi(function, search_params, std_threshold):
    """
    A simple multidimensional greedy search optimization function.
    
    :param function: The objective function to minimize.
    :param start_params: Dictionary of starting hyperparameters.
    :param N: Number of iterations to perform.
    :param u: Step size for exploring the hyperparameter space.
    :return: The optimized hyperparameters and the history of values.
    """

    start_params = {
        "Temperature": search_params["Temperature"][0],
        "Cooling_Schedule": search_params["Cooling_Schedule"][0],
        "N_Proposals": search_params["N_Proposals"][0]
    }
    
    x = start_params.copy()  # Current position in the hyperparameter space
    history = []  # Track history of positions

    #initalise
    y_x = function(**x, N_iterations= 5)

    while y_x > std_threshold:
        history.append(x.copy())


        xright = x.copy()        
        for param in x.keys():

            # check if max param reached
            if xright[param] >= search_params[param][1]:
                pass
            else:
                # Use the bespoke step size for each hyperparameter
                xright[param] += search_params[param][-1]
            
        yright = function(**xright, N_iterations = 3)
        print()
        print("std")
        print(yright)
        print()
        x = xright
        y_x = yright

    return x, y_x, history


def simulated_annealing_multivariate(
    model,
    search_space_real,
    data_feature_generator,
    data_feature_generator_args,
    show_progress_bar,
    SA_Hyper_Params
    ):
    
    from tqdm import tqdm  # Import tqdm for progress bar

    # fit the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_to_fit = np.array(search_space_real).T # transpose to get a 2d array of [[min][max]]
    scaler.fit(data_to_fit)


    def predict_scaled_proposal(
            scaled_data_point_np_array,
            scaler,
            model,
            data_feature_generator,
            data_feature_generator_args
            ):
        # rescale datapoint
        data_point_np_array = scaler.inverse_transform(
            scaled_data_point_np_array.reshape(1, -1) # rescale for single sample
            )
        # stick data point into df
        data_point_df = pd.DataFrame(
            data_point_np_array,
            columns=data_feature_generator_args[0]
            )
        # generate the feature matrix
        data_point_df_features = data_feature_generator(
            data_point_df,
            data_feature_generator_args[1]
            )
        # Reorder feature_matrix columns to match X_true_feature_matrix - the training data
        data_point_df_features = data_point_df_features[list(data_feature_generator_args[2])]
        # Predict the outputs using the trained model
        return model(data_point_df_features).to_numpy()
        
    def accept_proposal(current_value, proposal_value, T):
            """
            Determine whether to accept the new proposal based on the SA acceptance logic.

            Parameters:
            - current_value: The objective function value of the current solution.
            - proposal_value: The objective function value of the proposed solution.
            - T: The current temperature in the SA algorithm.

            Returns:
            - Boolean value indicating whether the proposal should be accepted.
            """
            # Calculate the change in the objective function value
            ΔE = proposal_value - current_value
            
            # If the proposal is better, accept it
            if ΔE > 0:
                return True
            else:
                # Adjust the argument for the exponential function to avoid overflow
                argument = ΔE / T
                # Threshold to avoid overflow, can be adjusted based on the maximum float value
                overflow_threshold = -np.log(np.finfo(float).max)
                
                if argument < overflow_threshold:
                    # If the argument is less than the threshold, P_accept is effectively 0
                    return False
                else:
                    P_accept = np.exp(argument)
                    random_number = np.random.rand()
                    return random_number < P_accept
    def SA(
        search_space,
        function,
        Temperature,
        Cooling_Schedule,
        N_Proposals,
        show_progress_bar = show_progress_bar
        ):

        

        scale = np.sqrt(Temperature)

        # Generate a random start point within the search space for each dimension
        start = np.array([np.random.uniform(low, high) for low, high in search_space])
        
        x = np.copy(start)

        current_best_y = function(
            x,
            scaler,
            model,
            data_feature_generator,
            data_feature_generator_args
            )

        history_x = [x]

        if show_progress_bar:
            for i in tqdm(range(N_Proposals), desc="Searching"):

                # sample proposalsimulated_annealing_multivariate
                proposal = x + np.random.normal(size=x.shape) * scale # uniform(-1, 1, size = x.shape) * scale

                # Check if proposal is within bounds for each dimension
                within_bounds = np.all([low <= prop <= high for prop, (low, high) in zip(proposal, search_space)])

                proposal_y = function(
                    proposal,
                    scaler,
                    model,
                    data_feature_generator,
                    data_feature_generator_args
                    )
                
                # if accepted
                if within_bounds and accept_proposal(current_best_y, proposal_y, Temperature):
                    x = proposal # update x
                    current_best_y = proposal_y # update new best y
                    Temperature *= Cooling_Schedule # reduce the temperature
                    history_x.append(x) # append accepted proposal to history
                    # Cooling down
                
                Temperature *= Cooling_Schedule
        else:
            for i in range(N_Proposals):

                # sample proposalsimulated_annealing_multivariate
                proposal = x + np.random.normal(size=x.shape) * scale # uniform(-1, 1, size = x.shape) * scale

                # Check if proposal is within bounds for each dimension
                within_bounds = np.all([low <= prop <= high for prop, (low, high) in zip(proposal, search_space)])

                proposal_y = function(
                    proposal,
                    scaler,
                    model,
                    data_feature_generator,
                    data_feature_generator_args
                    )
                
                # if accepted
                if within_bounds and accept_proposal(current_best_y, proposal_y, Temperature):
                    x = proposal # update x
                    current_best_y = proposal_y # update new best y
                    Temperature *= Cooling_Schedule # reduce the temperature
                    history_x.append(x) # append accepted proposal to history
                    # Cooling down
                
                Temperature *= Cooling_Schedule


        # get real values of final x
        final_max_x = scaler.inverse_transform(x.reshape(1, -1))

        return final_max_x, current_best_y, history_x


    # Define the search space for each dimension
    # Generate a search space of [-1, 1] for a specified number of variables
    search_space = [(-1, 1) for _ in range(len(data_feature_generator_args[0]))]  # for x1, x2, x3..

    final_max_x, final_max_y, history_x = SA(
        search_space,
        function = predict_scaled_proposal,
        Temperature = SA_Hyper_Params["Temperature"],
        Cooling_Schedule = SA_Hyper_Params["Cooling_Schedule"],
        N_Proposals = SA_Hyper_Params["N_Proposals"],
        show_progress_bar = show_progress_bar
        )

    return final_max_x, final_max_y, history_x



def predict_matrix_scaled_proposals(
        scaled_proposal_matrix,
        scaler,
        model,
        data_feature_generator,
        data_feature_generator_args
        ):
    # Rescale the proposals to their original scale
    proposals_np_array = scaler.inverse_transform(scaled_proposal_matrix)
    
    # Convert the proposals into a DataFrame
    proposals_df = pd.DataFrame(proposals_np_array, columns=data_feature_generator_args[0])
    
    # Generate the feature matrix for all proposals
    proposals_df_features = data_feature_generator(proposals_df, data_feature_generator_args[1])
    
    # Reorder the columns of the feature matrix to match the training data's feature matrix
    proposals_df_features = proposals_df_features[list(data_feature_generator_args[2])]
    
    # Predict the outputs for all proposals using the trained model
    predictions = model.predict(proposals_df_features).to_numpy()
    
    return predictions
    
def accept_proposal_matrix(
    current_y,
    proposal_y,
    current_x,
    proposal_x,
    T
    ):
    """
    Determine whether to accept the new proposal based on the SA acceptance logic.

    Parameters:
    - current_value: The objective function value of the current solution.
    - proposal_value: The objective function value of the proposed solution.
    - T: The current temperature in the SA algorithm.

    Returns:
    - Boolean value indicating whether the proposal should be accepted.
    """

    # Calculate the change in the objective function value
    ΔE = proposal_y - current_y

    # generate array containing the exponent: delta E divided by T
    argument = ΔE / T

    # logical or. set up to return false.
    accept_proposal_boolean = np.logical_or(

        # if delta is greater: True
        (ΔE > 0),

        # if either 
        np.logical_and(
            # invert result of argument < overflow_threshold - so true if argument is greater, false if lower which will make the whole result false
            # overflow threshold:  -np.log(np.finfo(float).max)
            np.logical_not(argument < -np.log(np.finfo(float).max)),
            
            # true if random_number < P_accept
            # generate array of sampled probabilities between 0 and 1.
            # generate array P_accept= e ^ delta E divided by T
            # generate boolean array that is true if sample is less than probability of acceptance
            (np.random.rand(proposal_y.shape[0]) < np.exp(np.clip(argument, None, np.log(np.finfo(float).max))))
            )
        )


    # Generate new current_y_new based on if the proposal is accepted or not.
    current_y_new = np.where(accept_proposal_boolean, proposal_y, current_y)
    
    # use the same to update current x
    # Initialize an empty array with the same shape as current_x to hold the new values
    current_x_new = np.empty_like(current_x)

    # Use boolean indexing for selection
    current_x_new[accept_proposal_boolean] = proposal_x[accept_proposal_boolean]
    current_x_new[~accept_proposal_boolean] = current_x[~accept_proposal_boolean]

    return current_y_new, current_x_new



def sim_anneal_matrix(
    model,
    search_space_real,
    data_feature_generator,
    data_feature_generator_args,
    show_progress_bar,
    SA_Hyper_Params
    ):

    # fit the scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_to_fit = np.array(search_space_real).T # transpose to get a 2d array of [[min][max]]
    scaler.fit(data_to_fit)

    # initialise temperature
    Temperature = SA_Hyper_Params["Temperature"]

    # Define the search space for each dimension
    # Generate a search space of [-1, 1] for a specified number of variables
    search_space = np.array([[-1, 1] for _ in range(len(data_feature_generator_args[0]))])  # for x1, x2, x3..

    # Generate the 2D matrix (particles, search_space)
    start = np.array([[np.random.uniform(low, high) for low, high in search_space] for _ in range(SA_Hyper_Params["particles"])])

    current_x = np.copy(start)
    current_y = predict_matrix_scaled_proposals(
            scaled_proposal_matrix = current_x,
            scaler = scaler,
            model = model,
            data_feature_generator = data_feature_generator,
            data_feature_generator_args = data_feature_generator_args
            )

    # initial history init
    history_y = current_y.copy()
    history_x = np.expand_dims(scaler.inverse_transform(current_x), axis=0) # expand dims to make 3d matrix with one layer

    from tqdm import tqdm  # Import tqdm for progress bar

    for i in tqdm(range(SA_Hyper_Params["N_Proposals"]), desc="Searching"):

        # sample proposalsimulated_annealing_multivariate
        proposal_x = np.random.uniform(low=-1, high=1, size=current_x.shape)

        # test
        proposal_y = predict_matrix_scaled_proposals(
                scaled_proposal_matrix = proposal_x,
                scaler = scaler,
                model = model,
                data_feature_generator = data_feature_generator,
                data_feature_generator_args = data_feature_generator_args
                )
        
        # Simulated annealing round
        current_y, current_x = accept_proposal_matrix(
            current_y,
            proposal_y,
            current_x,
            scaler.inverse_transform(proposal_x),
            T = Temperature
            )
        # Cooling down
        Temperature *= SA_Hyper_Params["Cooling_Schedule"]

        # update history
        history_y = np.vstack((history_y, current_y))

        history_x = np.concatenate((history_x, np.expand_dims(current_x.copy(), axis=0)), axis=0) # adds current x as another layer

    return history_y, history_x
