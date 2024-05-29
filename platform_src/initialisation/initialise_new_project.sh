#! /bin/bash



# Define functions for different actions
initialise_new_project() {

    local new_dir_name="$1"
    local template_project_path="$2"

    # initialise dir
    mkdir "/app/Projects/$new_dir_name"

    # copy
    cp -r $template_project_path "/app/Projects/$new_dir_name/"

    
}

executor="$3"

#### for executing in streamlit

# Check if new_dir_name is not set or is empty and then assign from $1
if [[ "$3" == "streamlit" ]]; then
    new_dir_name="$1"
    modeltype="$2"

    # Check modeltype and set template_project_path accordingly
    case "$modeltype" in
        "2 level Full Factorial")
            template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/2_level_Full_Factorial_Template/*"
            ;;
        "Plackett Burman")
            template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/Plackett_Burman/*"
            ;;
        "Central Composite Design")
            template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/Central_Composite_Design_Template/*"
            ;;
        "Full Factorial")
            template_project_path="/app/platform_src/initialisation/Templates/Project_Templates/Full_Factorial_Template/*"
            ;;
        *)

            exit 1
            ;;
    esac

    initialise_new_project "$new_dir_name" "$template_project_path"

else
    initialise_new_project "$new_dir_name" "$template_project_path"
fi


