# imports
import pandas as pd
from datetime import datetime, time
import json
"""
Data Cleaning and Tidying functions
"""

def baseline_subtract_data(melted_timecourse_data, trimmed_timecourse_data, well_metadata):


    # get the well types as lists
    # all wells
    well_list = []
    negative_control_well_list = []

    for well in well_metadata:
        well_list.append(well)
        if well_metadata[well]["Well_Type"] == "Negative_Control":
            negative_control_well_list.append(well)

    # now create column in trimmed_timecourse_data for negative control average
    trimmed_timecourse_data["negative_mean"] = trimmed_timecourse_data[negative_control_well_list].mean(axis=1)

    # get unique times
    time_list = list(melted_timecourse_data["Time"].unique())

    # list for populating
    RFUs_Baseline_Subtracted_List = []
    # slice by well
    for well in well_list:
        well_slice = melted_timecourse_data[melted_timecourse_data["Well"] == well].copy()
        #slice by time
        for time in time_list:
            # minus every well from the average of the negatives at that time point
            RFUs_Baseline_Subtracted = int(well_slice[well_slice["Time"]==time]["RFUs"]) - int(trimmed_timecourse_data[trimmed_timecourse_data["Time"]==time]["negative_mean"])
            RFUs_Baseline_Subtracted_List.append(RFUs_Baseline_Subtracted)
    # set list as new column
    melted_timecourse_data["RFUs_Baseline_Subtracted"] = RFUs_Baseline_Subtracted_List

    return melted_timecourse_data

def GetTimeInMinutes(trimmed_timecourse_data):

    # initialise list for populating
    seconds_list = []

    # interate over the rows to get the index
    for i, row in trimmed_timecourse_data.iterrows():
        
        # individually strip the hours, minutes and seconds out of trimmed_timecourse_data["Time"] by using the index i
        # multiply by relevant factor
        # append to list
        seconds_list.append(datetime.strptime(str(trimmed_timecourse_data["Time"][i]), "%H:%M:%S").second + (datetime.strptime(str(trimmed_timecourse_data["Time"][i]), "%H:%M:%S").minute * 60 ) + (datetime.strptime(str(trimmed_timecourse_data["Time"][i]), "%H:%M:%S").hour * 60 * 60))

    # insert the seconds list as Time
    trimmed_timecourse_data["Time"] = seconds_list
    # Divide by 60 to get minutes
    trimmed_timecourse_data["Time"] = trimmed_timecourse_data["Time"] / 60

    # if time starts at 0.05 or above 0, subtract the first time point from all
    if trimmed_timecourse_data["Time"][0] > 0:
        trimmed_timecourse_data["Time"] = trimmed_timecourse_data["Time"] - trimmed_timecourse_data["Time"][0]

    return trimmed_timecourse_data

def ExtractExperimentMetadataFromRaw(raw_metadata):
    """
    Extracts key experiment metadata from the raw data file by hardcoded slicing
    """

    experiment_metadata_dict = {

        # concatenates plate reader name to serial number
        "Plate_Reader_Protocol": raw_metadata.iloc[1,1][-18:-4],
        "Plate_Reader": raw_metadata.iloc[5,1] + "_" + str(raw_metadata.iloc[6,1]),
        # extracts the characters containing the number and converts to float
        "Reaction_Temp_C": float(raw_metadata.iloc[11,1][-4:-2]),
        "Gain": float(raw_metadata.iloc[20,1][-2:]),
        "Date": raw_metadata.iloc[3,1]
    }

    ## sanity check to confirm that the plate reader protocol stated is the same one recorded in the file.
    if experiment_metadata_dict["Plate_Reader_Protocol"] != "Timecourse_GFP":
        raise ValueError("The raw datafile was produced by a different plate reader protocol than Timecourse_GFP.")
    else:
        pass

    return experiment_metadata_dict




def TrimRawTimecourseData(raw_timecourse_data, well_metadata):
    """"
    Trimming raw timecourse data

    Gets rid of all surounding cells in the dataframe
    Sets columns
    Deletes temp column
    """

    ## Find the last row /time point. There are two possible senarios:
    # 1. that the full time course has been completed in which case look for Results in column 0 by strong matching
    # 2. that the timecourse was terminated early in which case look for the time point where time(0,0,0): 00:00:00
    # use it to calculate the last_data_row_index

    for i,row in raw_timecourse_data.iterrows():
        # if whole timecourse has been found
        # subtract one to account for spare row
        if row[0] == "Results":
            last_data_row_index = i-1
        
        # or if terminated early.
        # use i as the actual index.
        elif row[1] == time(0,0,0):
            last_data_row_index = i
            break

        else:
            last_data_row_index = raw_timecourse_data.shape[0]

    # slice the raw_data
    trimmed_timecourse_data = raw_timecourse_data.iloc[:last_data_row_index,1:]
    
    # set columns using top row
    trimmed_timecourse_data.columns = trimmed_timecourse_data.iloc[0,:]
    # delete row
    trimmed_timecourse_data = trimmed_timecourse_data.iloc[1:,:].reset_index(drop=True)
    # use the wells in well_metadata.keys() to trim columns
    column_list = ["Time"] + list(well_metadata.keys())
    # trim
    trimmed_timecourse_data = trimmed_timecourse_data[column_list]
    
    return trimmed_timecourse_data


def MeltDataByExperimentWells(trimmed_timecourse_data, well_metadata):
    
    melted_timecourse_data = pd.melt(
        
        # dataset
        trimmed_timecourse_data,
        
        # column not to be changed
        id_vars="Time",
        
        # columns to melt: the well names
        value_vars = list(well_metadata.keys()),
        
        # the new name for the column containing the melted column names
        var_name='Well',

        # The name for the column containing the melted values
        value_name='RFUs'
        )
    
    return melted_timecourse_data


"""
Metadata annotation functions
"""

def WellSpecificMetadataAnnotation(melted_timecourse_data, well_metadata):
    
    """
    Gets wells from well_metadata
    Slices df based on well.
    Populates columns with the metadata from that well
    Reassembles
    """

    # initialise empty df to be populated later.
    timecourse_annotated_wells = pd.DataFrame()

    # iterate over the wells in well_metadata
    for well in list(well_metadata.keys()):


        # slice dataframe to get one well
        well_specific_slice = melted_timecourse_data[melted_timecourse_data["Well"] == well].copy()

        # iterate over the metadata associated with that well and annotate accordingly
        for metadata_key, metadata_value in well_metadata[well].items():
            well_specific_slice[metadata_key] = metadata_value
            #well_specific_slice.loc[:,metadata_key] = metadata_value


        # add freshly annotated well_specific_slice to the new df
        timecourse_annotated_wells = pd.concat([timecourse_annotated_wells, well_specific_slice])
    

    return timecourse_annotated_wells


def AnnotateExperimentWideMetadata(timecourse_annotated_wells, experiment_metadata_dict):
    """
    Iterates over experiment_metadata_dict
    Populates columns with the metadata
    """
    # iterate over the metadata and annotate accordingly
    for metadata_key, metadata_value in experiment_metadata_dict.items():
        timecourse_annotated_wells.loc[:, metadata_key] = metadata_value

    return timecourse_annotated_wells


def WriteConditionStringColumn(timecourse_annotated, project_path):

    # extract experiment variables
    design_parameters_path = project_path + "/design_parameters.json"
    design_parameters_dict = json.load(open(design_parameters_path, 'r'))
    variables_dict = design_parameters_dict["Variables"]
    experiment_variables = list(variables_dict.keys())

    # slice the columns that contain the condition concentrations
    stringed_variables = timecourse_annotated[experiment_variables]

    # initialse an empty df to add the stringed columns to.
    string_df = pd.DataFrame()

    for i, column in enumerate(stringed_variables):

        # add the column name to the string of the concentration
        condition = stringed_variables[column].name + ": " + stringed_variables[column].astype(str)
        
        # add that new column to the df
        string_df = pd.concat([string_df, condition], axis = 1)

    # concatenate the strings and assign to the Condition column in timecourse_annotated
    timecourse_annotated["Condition"] = string_df[experiment_variables].agg(', '.join, axis=1)

    return timecourse_annotated
