#! /bin/bash

# Define functions for different actions
initialise_new_project() {


    # define design type
    echo -e "\n"
    echo -e "Choose design type: "
    echo -e "Screening Designs: "
    echo -e "1. 2 Level Full Factorial"
    echo -e "2. Plackett-Burman"
    echo -e "Response Surface Methodology Optimisation "
    echo -e "3. Central Composite Design"
    echo -e "4. Full Factorial"
    echo -e "5. Custom"

    # get user choice
    read -p "Enter choice [1-6]: " newdesignchoice


    # copy template across
    case $newdesignchoice in
        1)  template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/2_level_Full_Factorial_Template/*"
            ;;
        
        2)  template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/Plackett_Burman/*"
            ;;

        3)  template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/Central_Composite_Design_Template/*"
            ;;

        4)  template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/Full_Factorial_Template/*"
            ;;

        5) 
            echo "Custom"
            ;;

        *) 
            echo "Invalid choice, try again"
            ;;
    esac

    echo -e "\n"
    # get name
    read -p "Enter the name of the new Project: " new_dir_name


    # initialise dir
    mkdir "/app/Projects/$new_dir_name"

    # copy
    cp -r $template_project_path "/app/Projects/$new_dir_name/"

    
    echo -e "\n"
    echo -e "Initialised New Project: $new_dir_name"
    echo -e "\n"
    echo -e "Please configure the experimental parameters before generating your design."
    echo -e "\n"

}
