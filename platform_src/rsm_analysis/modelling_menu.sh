#!/bin/bash

source /app/platform_src/rsm_analysis/modelling_initialise_model_directory.sh
source /app/platform_src/rsm_analysis/modelling_specific_model_menu.sh


delete_model_directory() {
    # Sub-menu loop
    while true; do
        # Get a list of models and display as options
        modelsoptions=($(ls -d "$modelsdirectory"/*/))
        num_modelsoptions=${#modelsoptions[@]}
        echo -e "\n"
    
        for ((i=0; i<num_modelsoptions; i++)); do
            # extract just the basename
            model_name=$(basename "${modelsoptions[i]}")
            echo "[$((i+1))] $model_name"
        done
        read -p "Please select a model to delete: " model_to_delete_num

        # Check if input is a valid number
        if [[ $model_to_delete_num =~ ^[0-9]+$ ]] && [ $model_to_delete_num -ge 1 ] && [ $model_to_delete_num -le $num_modelsoptions ]; then
            # The user has chosen one of the directories
            model_to_delete_dir=${modelsoptions[$model_to_delete_num-1]}
            model_to_delete_name=$(basename "${modelsoptions[$model_to_delete_num-1]}")
            # Verification
            echo -e "You chose: $model_to_delete_name"
            read -p "Please enter the model name to confirm deletion: " user_input
            if [ "$user_input" == "$model_to_delete_name" ]; then
                rm -r "$model_to_delete_dir"
                echo -e "\n"
                echo -e "Deleted $model_to_delete_name"
                break
            else
                echo -e "Your Input didn't match the Model Name, exiting.."
                break
            fi
        else
            echo "Invalid choice. Please enter a number between 1 and $num_modelsoptions."
        fi
    done
}


modelling_menu() {

    # initialise directories
    # ModelsDirectory
    output="Output/"
    models="/models"
    modelsdirectory="$chosen_dir$output$feature_to_model$models"

    # Sub-menu loop
    while true; do

        # spacer
        echo -e "\n"
        echo -e "Modelling Menu."
        echo -e "\n"

        # Create an indexed array with directory names
        options=($(ls -d "$modelsdirectory"/*/))

        # Display options with indices
        echo "Select a model or action:"
        index=1
        for ((i=0; i<${#options[@]}; i++)); do
            model_name=$(basename "${options[i]}")
            echo "[$index] $model_name"
            ((index++))
        done

        # Add custom options
        #spacer
        echo -e "\n"
        echo -e "Manage Models:"
        echo -e "[$index] Create model"
        create_model_index=$index
        ((index++))

        echo -e "[$index] Delete model"
        delete_model_index=$index
        ((index++))
    
        echo -e "[$index] Return to Project Main Menu"
        return_index=$index


        # Read user choice
        read -p "Enter choice (1-$index): " choice

        # Validate and process the choice
        if [[ $choice =~ ^[0-9]+$ ]]; then # Check if input is a number

            if [ "$choice" -ge 1 ] && [ "$choice" -le "$index" ]; then # Now, check if within range
                
                # Check specific options within the valid range
                if [ "$choice" -le "${#options[@]}" ]; then # Assuming options is an array of valid choices
                    
                    # if model selected..
                    selected_option_index=$((choice - 1))
                    select_model_name=$(basename "${options[$selected_option_index]}")
                    select_model_dir=${options[$selected_option_index]}

                    specific_model_menu

                elif [ "$choice" -eq "$create_model_index" ]; then
                    initialise_model_directory

                elif [ "$choice" -eq "$delete_model_index" ]; then
                    delete_model_directory

                elif [ "$choice" -eq "$return_index" ]; then
                    break

                else
                    echo "Invalid selection"
                fi
            else
                echo "Invalid selection. Please choose a number between 1 and $index."
            fi
        else
            echo "Invalid input. Please enter a number."
        fi


    done
}

