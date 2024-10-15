#!/bin/bash

# Function to edit design_parameters JSON file
edit_well_specific() {

    local chosen_dir=$chosen_dir
    local filename="design_parameters.json"
    local path="$chosen_dir$filename"

    # Check if the file exists
    if [[ -f "$path" ]]; then
        # Open the file in a text editor, e.g., nano
        nano "$path"
    else
        echo "JSON file not found: $filename"
    fi
}

# Function to edit design metadata JSON file
edit_metadata() {

    local chosen_dir=$chosen_dir
    local filename="template_well_metadata.json"
    local path="$chosen_dir$filename"

    # Check if the file exists
    if [[ -f "$path" ]]; then
        # Open the file in a text editor, e.g., nano
        nano "$path"
    else
        echo "JSON file not found: $filename"
    fi
}

configure_experiment_parameters() {

    # spacer
    echo -e "\n"

    local chosen_dir=$chosen_dir

    # Sub-menu loop
    while true; do
        echo "What would you like to configure?:"
        echo "1. Input Variables & Design Parameters"
        echo "2. Metadata across all variables"
        echo "3. Return"
        read -p "Enter choice [1-3]: " configmenuchoice

        case $configmenuchoice in
            1) edit_well_specific;;
            2) edit_metadata;;
            3)  
                echo -e "\n"
                echo -e "Configuration complete."
                echo -e "Regenerate Design to implement changes."
                echo -e "Exiting."
                echo -e "\n"
                break # This exits the sub-menu loop
                ;;
            *) echo "Invalid choice, try again";;
        esac
    done

}

