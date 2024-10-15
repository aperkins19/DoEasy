#!/bin/bash

build_response_directory() {

    # initialise directories
    chosen_dir=$chosen_dir
    # Directory to check
    outputdir="Output/"
    directory="$chosen_dir$outputdir$feature_to_model"

    ## subdirectories
    model="/models"
    modeldirectory="$directory$model"
    exploratory="/explore"
    exploratorydirectory="$directory$exploratory"
    
    # Check if the directory exists
    directoryexists=false
    if [ -d "$directory" ]; then
        echo "Directory exists: $directory"
        directoryexists=true

        echo "Analysis appears to already exist for this metric. Would you like to delete progress & start again? (y/n)"
        read user_input

        # Convert the input to lowercase for case-insensitive comparison
        user_input_lowercase=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')
        if [[ "$user_input_lowercase" == "y" ]]; then
            
            rm -r $directory
            mkdir $directory $exploratorydirectory $modeldirectory

        else
            :
        fi

    else
        mkdir $directory $exploratorydirectory $modeldirectory
    fi
}

create_new_response_variable() {

    # spacer
    echo -e "\n"

    local chosen_dir=$chosen_dir
    local tidydata="Datasets/tidy_dataset.csv"
    local tidydatapath="$chosen_dir$tidydata"

    # check if tidy data exists
    # if not
    if ! [ -e "$tidydatapath" ]; then
        echo -e "No processed datasets present, can't select a response variable. Please resolve."
        
    else

        ## chose Y
        # Sub-menu loop
        while true; do

            # Get a list of directories and display as options
            # Read the first line of the CSV file into a variable
            #column_names=$(head -n 1 "$tidydatapath")
            IFS=',' read -ra column_names <<< "$(head -n 1 "$tidydatapath")"

            num_options=${#column_names[@]}

            # Project Selection Menu
            for ((i=0; i<num_options; i++)); do
                echo "[$((i+1))] ${column_names[i]}"
            done

            echo -e "\n"
            echo -e "These are the diamensions in the tidy dataset currently resident in the project."
            echo -e "Please select which you would like to use as the response variable in this analysis."
            # Read user's choice
            read -p "Enter your choice: " choice

            # Validate the choice is a number and within range
            if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= num_options )); then
                # The user has chosen one of the columns
                feature_to_model=${column_names[choice-1]}
                echo -e "\n"
                echo -e "You chose: $feature_to_model"
                # build function
                build_response_directory
                echo -e "$feature_to_model initialised"
                echo -e "\n"

                break
            else
                echo "Invalid choice. Please enter a number between 1 and $num_options."
                echo -e "\n"
            fi
        done

        ###########################
        # check if directory exists
    
    fi
}

