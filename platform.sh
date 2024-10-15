#!/bin/bash

## import
source /app/platform_src/select_project.sh
source /app/platform_src/initialisation/initialise_new_project.sh
source /app/platform_src/initialisation/copy_example_project.sh



# Main menu loop
while true; do
    echo -e "\n"
    echo "Main Menu:"
    echo "1. Open existing project"
    echo "2. Create New Project"
    echo "3. Explore Example Project"
    echo "4. Exit"
    read -p "Enter choice [1-3]: " choice

    case $choice in
        1) select_project;;
        2) initialise_new_project;;
        3) copy_example_project;;
        4) echo "Exiting script."; exit 0;;

        *) echo "Invalid choice, try again.";;
    esac
done