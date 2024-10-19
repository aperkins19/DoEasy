import streamlit as st
import subprocess
import pandas as pd 
import numpy as np 
import os
import shutil
import json
import time
import base64
from PIL import Image
import io
import pickle 

import sys

##################################################################
# Initialize the Experiment_List variable if it does not exist
def get_subdirectories(directory):
    # List subdirectories in the given directory. Return None if no subdirectories are found
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    return subdirs if subdirs else None


# Create a function to display the table
def display_experiment_table(container):
    # List of experiments
    experiment_list = st.session_state["Experiment_List"]
    
    # If there are experiments, display them in a table format
    if experiment_list:
        # Prepare data as a list of dictionaries or tuples
        data = []
        for idx, experiment in enumerate(experiment_list):


            experiment_path = f"/app/Projects/{experiment}/"
            # Load the JSON data from the file into a Python dictionary
            with open(experiment_path+"design_parameters.json", 'r') as json_file:
                design_params = json.load(json_file)

            # You can add more columns here as needed
            data.append({
                "Experiment Name": experiment,
                "Type": design_params["Design_Type"]
            })

            del design_params
        
        # Convert the data into a DataFrame for easy table rendering
        df = pd.DataFrame(data)
        
        # Display the table
        container.table(df)
    else:
        container.write("No experiments available to display.")




# Function to execute a Bash script to create new experiment
def execute_create_experiment_script(name, model_type, container):

    if model_type == "2 Level Full Factorial":
        model_type_string = "2_level_Full_Factorial_Template"
    elif model_type == "Plackett-Burman":
        model_type_string = "Plackett_Burman"
    elif model_type == "Central Composite Design":
        model_type_string = "Central_Composite_Design_Template"
    elif model_type == "Full Factorial":
        model_type_string = "Full_Factorial_Template"
        
    # copy template into new project directory
    template_project_path=f"/app/platform_src/initialisation/Templates/Project_Templates/{model_type_string}"
    new_project_directory=f"/app/Projects/{name}"

    # Step 1: tCreate the directory if it doesn't exist
    try:
        os.makedirs(new_project_directory, exist_ok=True)  # 'exist_ok=True' won't raise an error if the directory exists
        print(f"Project: {name} created successfully.")
    except OSError as error:
        print(f"Error creating Project: {error}")

    # Step 2: Copy the contents of the template directory
    try:
        shutil.copytree(template_project_path, new_project_directory, dirs_exist_ok=True)  # dirs_exist_ok=True to overwrite if needed
        st.success(f"{name} created. Please refresh the page.")
    except Exception as error:
        print(f"Error copying files: {error}")

    #st.rerun()
    container.success(f"Experiment Created: {new_experiment_name}. Please refresh the page to find the newly created Experiment.")







# Function to execute a Bash script to create new experiment
def execute_copy_experiment_script(name, experiment_to_copied, container):

    experiment_to_copied_path = f"/app/Projects/{experiment_to_copied}"
    new_experiment_path = f"/app/Projects/{name}"
    try:
        shutil.copytree(experiment_to_copied_path, new_experiment_path, dirs_exist_ok=True)  # dirs_exist_ok=True to overwrite if needed
        st.success(f"{name} created. Please refresh the page.")
    except Exception as error:
        print(f"Error copying files: {error}")




def create_experiment_form(container):
    # Start the app with a header

    # Create a form and add widgets
    create_new_form = container.form("my_form")
    create_new_form.subheader("Create Experiment:")
    new_experiment_name = create_new_form.text_input("Enter the Experiment Name:")

    # Perform validation as user types
    if new_experiment_name:
        if not st.session_state['Experiment_List'] is None:

            if new_experiment_name in st.session_state['Experiment_List']:
                container.error("This experiment name already exists. Please choose a different name.")
    new_experiment_type = create_new_form.selectbox("Choose Design Type:",
                            [
                                '2 Level Full Factorial',
                                'Plackett-Burman',
                                'Central Composite Design',
                                'Full Factorial'
                            ])
    submitted = create_new_form.form_submit_button("Submit")
    if submitted:
        # Call the execute_script function when the form is submitted
        result = execute_create_experiment_script(
            name = new_experiment_name,
            model_type = new_experiment_type,
            container = container
            )




def copy_experiment_form(container):
    # Start the app with a header
    copy_project_form = container.form("copy_project_form")

    copy_project_form.subheader("Copy Experiment:")

    experiment_to_copied = copy_project_form.selectbox("Select the experiment to copy:",
                            st.session_state["Experiment_List"])

    new_copy_experiment_name = copy_project_form.text_input("Enter the new name:")

    # Perform validation as user types
    if new_copy_experiment_name:
        if new_copy_experiment_name in st.session_state['Experiment_List']:
            container.error("This experiment name already exists. Please choose a different name.")

    submitted = copy_project_form.form_submit_button("Submit")
    if submitted:
        # Call the execute_script function when the form is submitted
        result = execute_copy_experiment_script(
            name = new_copy_experiment_name,
            experiment_to_copied = experiment_to_copied,
            container = container
            )


def delete_experiment_form(container):
    # Start the app with a header
    delete_project_form = container.form("delete_project_form")

    # Create a form and add widgets
    delete_project_form.subheader("Delete Experiment:")

    experiment_to_delete = delete_project_form.selectbox("Select the experiment to delete:",
                            st.session_state["Experiment_List"])

    experiment_to_delete_typed = delete_project_form.text_input("Type the name to confirm deletion:")

    
    submitted = delete_project_form.form_submit_button("Submit")
    if submitted:
        # Perform validation as user types
        if experiment_to_delete_typed == experiment_to_delete:
            try:
                shutil.rmtree(f"/app/Projects/{experiment_to_delete}")
                container.warning(f"{experiment_to_delete} deleted successfully. Refresh the page.")
            except Exception as e:
                container.error(f"{experiment_to_delete} failed to delete.")

#################################################################################################

if 'Experiment_List' not in st.session_state:
    st.session_state['Experiment_List'] = get_subdirectories("Projects/")


# Initialize the Current_Experiment variable if it does not exist
if 'Current_Experiment' not in st.session_state:
    st.session_state['Current_Experiment'] = None


##################################################################

st.header('DoEasy')



#### First check if experiments exist
if st.session_state["Experiment_List"] == ["Create New"]:
    st.subheader("No Experiments yet exist.")
    create_new = st.container(border=True)
    create_experiment_form(container=create_new)
else:

    experiment_table_container = st.container()
    experiment_table_expander = st.expander(
        "Experiments:",
        expanded=True)
    display_experiment_table(container = experiment_table_expander)

    row1 = st.columns(2)
    row2 = st.columns(2)

    # create new experiment
    create_new_project = row1[0].container()
    #create_new_title = row1[0].container(height=120)
    create_experiment_form(container=create_new_project)


    ######### copy experiment
    copy_project_container = row1[1].container()
    copy_experiment_form(container=copy_project_container)

    ######### delete experiment
    delete_project_container = row2[0].container()
    delete_experiment_form(container=delete_project_container)