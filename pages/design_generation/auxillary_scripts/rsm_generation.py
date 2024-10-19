import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from scipy.stats import qmc
import math 
import pandas as pd


def generate_center_point(num_dimensions: int, num_center_points:int, num_replicates:int):
    """
    Generate the central point with coordinates (0, 0, ..., 0) in an n-dimensional space.

    Parameters:
    num_dimensions (int): Number of dimensions.

    Returns:
    numpy.ndarray: Central point coordinates in an n-dimensional space.
    """

    # Create an n-dimensional array with zeros
    center_points = np.vstack([np.zeros(num_dimensions)]*num_center_points*num_replicates)

    return center_points

def generate_corner_points(num_dimensions: int, num_replicates:int):
    """
    Generate factorial points for an n-dimensional space based on the given levels.

    Parameters:
    levels (list): List of levels for each dimension.

    Returns:
    numpy.ndarray: Factorial points in an n-dimensional space.
    """

    num_levels = [2] * num_dimensions

    # Generate a grid of points for each dimension
    grids = [np.linspace(-1, 1, l) for l in num_levels]

    # Create the factorial points using meshgrid
    factorial_points = np.array(np.meshgrid(*grids)).T.reshape(-1, len(num_levels))

    if num_replicates > 1:
        replicates_list = [factorial_points]*num_replicates
        factorial_points = np.vstack(replicates_list)

        for i in range(0, num_dimensions,1):

            if i == 0:
                factorial_points = factorial_points[factorial_points[:, i].argsort()]
            else:
                factorial_points = factorial_points[factorial_points[:, i].argsort(kind='mergesort')]

    return factorial_points


def generate_axial_points(alpha:float, num_dimensions:int, num_replicates:int):

    # init central point
    center_point = np.zeros(num_dimensions)

    axial_points = []

    for i,dim in enumerate(center_point):

        upper_array = center_point.copy()
        lower_array = center_point.copy()

        upper_array[i] = alpha
        lower_array[i] = -alpha

        axial_points.append(upper_array)
        axial_points.append(lower_array)

    if num_replicates > 1:
        replicates_list = [axial_points]*num_replicates
        axial_points = np.vstack(replicates_list)

        for i in range(0, num_dimensions,1):

            if i == 0:
                axial_points = axial_points[axial_points[:, i].argsort()]
            else:
                axial_points = axial_points[axial_points[:, i].argsort(kind='mergesort')]

    #axial_points = np.vstack(axial_points)

    return axial_points



def generate_full_factorial(variables_dict: dict):
    import itertools
    """
    Generate full factorial points for an n-dimensional space based on the given levels.

    Parameters:
    levels (list): List of levels for each dimension.

    Returns:
    numpy.ndarray: Factorial points in an n-dimensional space.
    """

    # Generate levels for each variable using np.linspace
    for var, info in variables_dict.items():
        min_val = info["Min"]
        max_val = info["Max"]
        levels = info["Levels"]

        # if is categorical... quick and dirty fix
        if isinstance(max_val, str):
            variables_dict[var]["GeneratedLevels"] = np.array([min_val, max_val])
        else:
            variables_dict[var]["GeneratedLevels"] = np.linspace(min_val, max_val, levels)

    # Dynamically extract generated levels for all variables
    all_levels = [info["GeneratedLevels"] for info in variables_dict.values()]

    # Dynamically generate all combinations of levels for the full factorial design
    design = list(itertools.product(*all_levels))

    # build df
    design_df = pd.DataFrame(design, columns=list(variables_dict.keys()))

    # coded design
    coded_df = pd.DataFrame()
    # Dynamic mapping
    for col in design_df.columns:
        max_val = variables_dict[col]['Max']
        min_val = variables_dict[col]['Min']
        # Apply dynamic mapping for each value in the column
        coded_df[col] = design_df[col].apply(lambda x: "+" if x == max_val else ("-" if x == min_val else x))

    return coded_df


def generate_latin_hypercube_samples(
    num_dimensions:int,
    num_samples:int,
    num_replicates:int):


    # generate samples
    # the seed makes this deterministic - repeatable
    sampler = qmc.LatinHypercube(d=num_dimensions, seed = 123)
    sample = sampler.random(n=num_samples)





    if num_replicates > 1:
        replicates_list = [sample]*num_replicates
        sample = np.vstack(replicates_list)

        for i in range(0, num_dimensions,1):

            if i == 0:
                sample = sample[sample[:, i].argsort()]
            else:
                sample = sample[sample[:, i].argsort(kind='mergesort')]

    # scale samples for a coded design
    Upper_Bounds = np.array([1]*num_dimensions)  # Upper bounds for each dimension
    Lower_Bounds = np.array([-1]*num_dimensions)  # Lower bounds for each dimension

    rescaled_samples = Lower_Bounds + sample * (Upper_Bounds - Lower_Bounds)

    # 2 decimal places
    rescaled_samples = np.around(rescaled_samples, decimals=2)

    return rescaled_samples


def CentralCompositeDesign(
    num_dimensions:int,
    alpha:float,
    num_center_points:int,
    variable_names:list,
    num_replicates:int
    ):

    corner_points = pd.DataFrame(
        data = generate_corner_points(
            num_dimensions = num_dimensions,
            num_replicates = num_replicates
            ),

        columns = variable_names
        )
    corner_points["DataPointType"] = "Corner"
    
    center_points = pd.DataFrame(
        data = generate_center_point(
            num_dimensions = num_dimensions,
            num_center_points=num_center_points,
            num_replicates = num_replicates
            ),
        columns = variable_names
        )
    center_points["DataPointType"] = "Center"

    axial_points = pd.DataFrame(
        data = generate_axial_points(
            num_dimensions = num_dimensions,
            alpha=alpha,
            num_replicates = num_replicates
            ),
        columns = variable_names
        )
    axial_points["DataPointType"] = "Axial"

    CCD = pd.concat([corner_points, center_points, axial_points])

    return CCD

def GeneratePlackettBurman(
    num_dimensions:int,
    variable_names:list,
    num_replicates:int
    ):

    # https://books.google.co.uk/books?id=UgINAwAAQBAJ&pg=PA32&dq=Plackett-Burman+Design&hl=en&sa=X&redir_esc=y#v=onepage&q=Plackett-Burman%20Design&f=false
    # https://www.statisticshowto.com/plackett-burman-design/
    generation_vectors_dict = {

        8 : ["+", "+", "+", "-", "+", "-", "-"],
        12 : ["+", "+", "-", "+", "+", "+", "-", "-", "-", "+", "-"],
        16 : ["+", "+", "+", "+", "-", "+", "-", "+", "+", "-", "-", "+", "-", "-", "-"],
        20 : ["+", "+", "-", "-", "+", "+", "+", "+", "-", "+", "-", "+", "-", "-", "-", "-", "+", "+", "-"],
        24 : ["+", "+", "+", "+", "+", "-", "+", "-", "+", "+", "-", "-", "+", "+", "-", "-", "+", "-", "+", "-", "-", "-", "-"],
        36 : ["-", "+", "-", "+", "+", "+", "-","-", "-", "+", "+", "+", "+", "+", "-", "+", "+", "+", "-", "-", "+", "-", "-", "-", "-", "+", "-", "+", "-", "+", "+", "-", "-", "+", "-"]
    }


    # select vector
    if num_dimensions < 4:
        raise Exception("The number of variables is less than 4, therefore a Plackett-Burman design is not appropriate. If may be that a screening design is not required. Alternatively please choose instead a simple screening design or include more variables.")
    
    elif (num_dimensions < 8):
        selected_vector = generation_vectors_dict[8]

    elif (num_dimensions < 12):
        selected_vector = generation_vectors_dict[12]
    
    elif (num_dimensions < 16):
        selected_vector = generation_vectors_dict[16]
    
    elif (num_dimensions < 20):
        selected_vector = generation_vectors_dict[20]
    
    elif (num_dimensions < 24):
        selected_vector = generation_vectors_dict[24]
    
    elif (num_dimensions < 36):
        selected_vector = generation_vectors_dict[36]


    # initialise empty df for population
    coded_design_df = pd.DataFrame()

    # initialise working vector
    working_vector = selected_vector.copy()

    # construct full design
    for i in range(0, num_dimensions, 1):

        # copy and append "-" to construct final vector
        final_vector = working_vector.copy()
        final_vector.append("-")

        # append final vector to the df under the column name of the current variable name
        coded_design_df[variable_names[i]] = final_vector

        # now shuffle along the working vector (doesn't matter after the last one)
        working_vector.insert(0, working_vector[-1])
        working_vector.pop(-1)


    return coded_design_df

