#! /bin/bash

reset_predict_dir() {

    echo -e "\n"
    echo -e "WARNING - Deleting your progress is perminent."
    echo  "Make sure to copy any plots or output that you want to keep before continuing."
    echo  "The model, it's trained weights and parameters will not be affected."
    echo -e "\n"
    read -p "Type 'Yes I am sure' to confirm the reset: " user_input
    echo -e "\n"
    project_destination="${select_model_dir%/}/prediction/"

    # Check if the user input matches the confirmation string
    if [ "$user_input" == "Yes I am sure" ]; then
        # Delete the existing directory if it exists
        if [ -d "$project_destination" ]; then
            rm -rf "$project_destination"
        fi
        # Create the directory
        mkdir -p "$project_destination"

        echo "The directory has been reset."
    else
        echo "Reset cancelled. You didn't enter the correct text."
    fi


}