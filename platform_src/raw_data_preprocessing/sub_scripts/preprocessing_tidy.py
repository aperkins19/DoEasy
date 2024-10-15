"""
# Intro

First script in the data preprocessing that:
* Takes in the raw plate reader csv / excel file and well metadata file
* Returns Tidy Dataset annotated with the metadata.
"""

# imports
import pandas as pd


# functions
from sub_scripts.analysis_functions.preprocessing import *

def preprocessing_tidy(raw_data, well_metadata, project_path, negative_control_designated):


    """
    Slice and Tidy the raw data and metadata
    First deal with the metadata data contained in the raw data file.
    Then trim the raw data
    """

    ### file metadata

    # slice
    raw_metadata = raw_data.iloc[3:28, 0:2]
    # extract
    experiment_metadata_dict = ExtractExperimentMetadataFromRaw(raw_metadata)


    if raw_data[raw_data.columns[1]].eq("12:00:03").any():
        mask = raw_data[raw_data.columns[1]].eq("12:00:03")
        first_zero_time_index = mask.idxmax()

    else:

        # Step 1: Identify the first row where the second column has the value "00:00:00" starting from row 52
        # Note: Adjust the column index if your 'time' is not the second column (python uses 0-based indexing)
        first_zero_time_index = raw_data.iloc[51:][raw_data.columns[1]].eq("00:00:00").idxmax()

        # Step 2: Check if there actually is a "00:00:00" in the remaining rows; if not, use the entire DataFrame length
        if raw_data.iloc[first_zero_time_index][raw_data.columns[1]] != "00:00:00":
            first_zero_time_index = len(raw_data)

    # Step 3: Slice the DataFrame from row 52 up to the first row with "00:00:00" in the second column
    raw_timecourse_data = raw_data.iloc[51:first_zero_time_index, 0:].reset_index(drop=True)

    print(raw_timecourse_data)

    # # get all rows below the metadata
    # raw_timecourse_data = raw_data.iloc[51:,0:].reset_index(drop=True)

    # trim
    trimmed_timecourse_data = TrimRawTimecourseData(raw_timecourse_data, well_metadata)


    # convert date.time hh:mm:ss to mins
    trimmed_timecourse_data = GetTimeInMinutes(trimmed_timecourse_data)


    # Melt wellwise
    melted_timecourse_data = MeltDataByExperimentWells(trimmed_timecourse_data, well_metadata) 

    # baseline subtract
    if negative_control_designated == True:
        melted_timecourse_data = baseline_subtract_data(melted_timecourse_data, trimmed_timecourse_data, well_metadata)

    """
    Metadata Annotation
    """

    # annotate the metadata for each well
    timecourse_annotated_wells = WellSpecificMetadataAnnotation(melted_timecourse_data, well_metadata)

    # annotate the metadata covering the whole experiment
    timecourse_annotated = AnnotateExperimentWideMetadata(timecourse_annotated_wells, experiment_metadata_dict)

    # write condition as strings in Condition Column
    timecourse_annotated = WriteConditionStringColumn(timecourse_annotated, project_path)
    
    return timecourse_annotated
