#!/bin/bash

## import
source /app/platform_src/project_menu.sh

select_project() {

    # spacer
    echo -e "\n"

    # Sub-menu loop
    while true; do
        # Get a list of directories and display as options
        options=($(ls -d /app/Projects/*/))
        num_options=${#options[@]}

        # Project Selection Menu
        echo -e "\nPlease choose a Project (enter the number):\n"
        for ((i=0; i<num_options; i++)); do
            # get and display basename
            project_name=$(basename "${options[i]}")
            echo "[$((i+1))] $project_name"
        done

        ## return
        echo -e "[$(($num_options + 1))] return to main menu"

        # Read user's choice
        read -p "Enter your choice: " choice


        case $choice in
            *)
                if [[ $choice -ge 1 && $choice -le $num_options ]]; then
                    # The user has chosen one of the directories
                    chosen_dir=${options[$choice-1]}
                    chosen_project_name=$(basename "$chosen_dir")
                    echo "You chose: $chosen_project_name"
                    # Perform any action you want with the chosen directory here
                    project_menu
                elif [[ $choice -eq $((num_options + 1)) ]]; then
                    break
                else
                    echo "Invalid choice. Please enter a number between 1 and $((num_options + 1))."
                fi
                ;;
        esac

    done

}

