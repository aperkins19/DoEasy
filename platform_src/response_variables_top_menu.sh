#!/bin/bash

source /app/platform_src/rsm_analysis/rsm_analysis_menu.sh
source /app/platform_src/create_new_response_variable.sh


delete_response_variable() {
    echo -e "\n"
    echo -e "delete response variable function"
}

response_variables_top_menu() {

    local chosen_dir=$chosen_dir
    local output="Output/"
    local reponsevariablespath="$chosen_dir$output"
    
    # spacer
    echo -e "\n"
    echo -e "Analysis is performed with respect to a specfic response variables."
    echo -e "Choose an existing reponse variable:"
    echo -e "\n"

    # Sub-menu loop
    while true; do

        # Get a list of directories and display as options
        options=($(ls -d "$reponsevariablespath"*/))
        num_options=${#options[@]}

        # display options and their indicies:
        index=1
        for ((i=0; i<${#options[@]}; i++)); do
            reponse_variable_name=$(basename "${options[i]}")
            echo "[$index] $reponse_variable_name"
            ((index++))
        done

        # Add admin options
        #spacer
        echo -e "\n"
        echo -e "Manage:"
        echo -e "[$index] New Response Variable"
        create_response_variable_index=$index
        ((index++))

        echo -e "[$index] Delete Response Variable"
        delete_response_variable_index=$index
        ((index++))
    
        echo -e "[$index] Return to Project Main Menu"
        return_response_variable_index=$index

        echo -e "\n"

        # Read user choice
        read -p "Enter choice (1-$index): " choice

        # Validate and process the choice
        if [[ $choice =~ ^[0-$index]+$ ]]; then

            if [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
                
                # if model selected..

                # The user has chosen one of the directories
                feature_to_model=$(basename "${options[$choice-1]}")
                echo "You chose: $feature_to_model"
                # Perform any action you want with the chosen directory here
                rsm_analysis_menu
                

            # create model
            elif [ "$choice" -eq "$create_response_variable_index" ]; then
                echo "create"
                create_new_response_variable

            # delete model
            elif [ "$choice" -eq "$delete_response_variable_index" ]; then
                echo "delete"
                delete_response_variable

            # return
            elif [ "$choice" -eq "$return_response_variable_index" ]; then
                break

            else
                echo "Invalid selection"
            fi
        else
            echo "Invalid input. Please enter a number."
        fi

    done
}

