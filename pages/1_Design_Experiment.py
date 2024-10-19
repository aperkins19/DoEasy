import streamlit as st
import subprocess
import pandas as pd 
import numpy as np 
import os
import json

import base64
from PIL import Image
import io
import pickle 

import sys

#######################################
# Initialize the Experiment_List variable if it does not exist
def get_subdirectories(directory):
    # List subdirectories in the given directory. Return None if no subdirectories are found
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirs if subdirs else None



def DefineExperimentDesignParams(design_params, container):

    design_params_form = container.form("design_params_form")
    design_params_form.write("Define design parameters:")


    row1 = design_params_form.columns(1)
    row2 = design_params_form.columns(3)
    row3 = design_params_form.columns(3)
    row4 = design_params_form.columns(3)
    row5 = design_params_form.columns(3)

    # Add form fields
    name = row1[0].text_input("Design Name")
    number_of_centerpoints = row2[0].number_input("Number of Centerpoints", min_value=0, max_value=None)
    number_of_replicates = row2[1].number_input(
        "Number of Replicates",
        min_value=0,
        max_value=None,
        value=design_params["Technical_Replicates"]
        )

    if_bespoke_alpha = row3[0].toggle("Use Bespoke Alpha")
    bespoke_alpha = row3[1].number_input("Alpha Value", min_value=0, max_value=None)

    if_validation = row4[0].toggle(
        "Include Validation Points",
        value=design_params["Generate_Validation_Points"]
        )
    validation_ratio = row4[1].number_input("Validation Points", min_value=0, max_value=None)
    
    randomise = row5[0].toggle(
        "Randomise",
        value=design_params["Randomise"]
        )

    ##### variables

    design_params_form.write("Input Variables")
    # dyamically produce a form with the correct number of variables
    for i,var in enumerate(design_params["Variables"]):

        var_cols = design_params_form.columns(4)

        var_name = var_cols[0].text_input(
            "Variable Name",
            value = list(design_params["Variables"].keys())[i],
            key = "Input"+var+"name"
            )

        var_min = var_cols[1].text_input(
            "Min",
            value = design_params["Variables"][var]["Min"],
            key = "Input"+var+"min"
            )

        var_max = var_cols[2].text_input(
            "Max",
            value = design_params["Variables"][var]["Max"],
            key = "Input"+var+"max"
            )
        var_units = var_cols[3].text_input(
            "Units",
            value = design_params["Variables"][var]["Units"],
            key = "Input"+var+"untis"
            )

    ## Response Varibles
    design_params_form.write("Response Variables")
    # dyamically produce a form with the correct number of variables
    for i,var in enumerate(design_params["Response_Variables"]):

        var_cols = design_params_form.columns(4)

        var_name = var_cols[0].text_input(
            "Variable Name",
            value = list(design_params["Response_Variables"].keys())[i],
            key = "Response"+var+"name"
            )

        var_units = var_cols[1].text_input(
            "Units",
            value = design_params["Response_Variables"][var]["Units"],
            key = "Response"+var+"units"
            )

    # Add a submit button
    submitted = design_params_form.form_submit_button("Generate Design")

    # Handle form submission
    if submitted:
        st.write(f"Design Name: {name}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {gender}")
        if agreement:
            st.write("Thank you for agreeing to the terms and conditions!")
        else:
            st.write("Please agree to the terms and conditions to proceed.")

    # if Design_Type == "CCD":
    #     container.write("CCD")

    # if Design_Type == "2_Level_Full_Factorial":
    #     container.write("2_Level_Full_Factorial")

    # if Design_Type == "Plackett_Burman":
    #     container.write("Plackett_Burman")

    # if Design_Type == "Full_Factorial":
    #     container.write("Full_Factorial")


######################################

if 'Experiment_List' not in st.session_state:
    st.session_state['Experiment_List'] = get_subdirectories("Projects/")


# Initialize the Current_Experiment variable if it does not exist
if 'Current_Experiment' not in st.session_state:
    st.session_state['Current_Experiment'] = st.session_state['Experiment_List'][0]

######################################
########## side bar
# Using object notation
st.session_state['Current_Experiment'] = st.sidebar.selectbox(
    "Select an experiment:",
    st.session_state['Experiment_List'],
    index = st.session_state['Experiment_List'].index(st.session_state['Current_Experiment'])
    )

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )


#####################################

if st.session_state['Current_Experiment'] == None:
    st.warning("Create an experiment in the homepage to continue.")
    st.stop()
else:
    pass

experiment_path = f"/app/Projects/{st.session_state['Current_Experiment']}/"
# Load the JSON data from the file into a Python dictionary
with open(experiment_path+"design_parameters.json", 'r') as json_file:
    design_params = json.load(json_file)


design_experiment_container = st.expander(label="Design Experiment",expanded=True)


DefineExperimentDesignParams(
    design_params = design_params,
    container=design_experiment_container
    )


design_experiment_container.write(design_params["Design_Type"])