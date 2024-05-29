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

##################################################################
# Initialize the Experiment_List variable if it does not exist
def get_subdirectories(directory):
    # List subdirectories in the given directory. Return None if no subdirectories are found
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))] + ["Create New"]
    return subdirs if subdirs else ["Create New"]



# Function to execute a Bash script to create new experiment
def execute_create_experiment_script(name, model_type):
    # Prepare the command to execute
    # For example, a script that takes two arguments: name and model_type type
    command = ['/app/platform_src/initialisation/initialise_new_project.sh', name, model_type, "streamlit"]

    # Run the command and capture the output
    try:
        output = subprocess.run(command, check=True, text=True, capture_output=True)
        return output.stdout
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"



def create_experiment_form():
    # Start the app with a header
    st.subheader("Create Experiment:")

    # Create a form and add widgets
    with st.form("my_form"):
        new_experiment_name = st.text_input("Enter the Experiment Name:")

        new_experiment_type = st.selectbox("Choose Design Type:",
                                    [   '2 Level Full Factorial',
                                        'Plackett-Burman',
                                        'Central Composite Design',
                                        'Full Factorial'
                                    ])

        # Every form must have a submit button
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Call the execute_script function when the form is submitted
            result = execute_create_experiment_script(name = new_experiment_name, model_type = new_experiment_type)
            st.write(f"Experiment Created: {new_experiment_name}")
            st.write(f"Please refresh the page to find the newly created Experiment")


#################################################################################################

if 'Experiment_List' not in st.session_state:
    st.session_state['Experiment_List'] = get_subdirectories("Projects/")


# Initialize the Current_Experiment variable if it does not exist
if 'Current_Experiment' not in st.session_state:
    st.session_state['Current_Experiment'] = None


##################################################################



#### First check if experiments exist
if st.session_state["Experiment_List"] == ["Create New"]:
    st.subheader("No Experiments yet exist.")
    create_experiment_form()
else:
    st.header('Experiments:')
    model_type = st.selectbox(
        "Select Experiment to work with:",
        (st.session_state["Experiment_List"])
        )

    if model_type == "Create New":
        create_experiment_form()
    else:
        st.write('You selected:', model_type)
