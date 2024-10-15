#! /bin/bash

# Define functions for different actions
copy_example_project() {


    # define design type
    echo -e "\n"
    echo -e "Choose Example Project: "
    echo -e "Screening Designs: "
    echo -e "2. Plackett-Burman - PCR Optimisation"
    echo -e "Response Surface Methodology Optimisation "
    echo -e "3. Central Composite Design - PCR Optimisation"

    # get user choice
    read -p "Enter choice [1-6]: " exampleprojectchoice


    # copy template across
    case $exampleprojectchoice in

        
        2)  example_project_path="/app/platform_src/initialisation/Examples/Plackett_Burman_PCR/*"
            ;;

        3)  example_project_path="/app/platform_src/initialisation/Examples/Central_Composite_Design_PCR/*"
            ;;


        *) 
            echo "Invalid choice, try again"
            ;;
    esac
    # get name
    echo -e "\n"
    read -p "Enter the name of the new Project: " new_dir_name

    # initialise dir
    mkdir "/app/Projects/$new_dir_name"

    # copy
    cp -r $example_project_path "/app/Projects/$new_dir_name/"

    
    echo -e "\n"
    echo -e "Initialised Example Project: $new_dir_name"
    echo -e "You can find it in Open Existing Project"
    echo -e "\n"

}
