#!/bin/bash

generate_design() {

    local chosen_dir=$chosen_dir
    
    echo -e "\n"

    ##########################################################################
    # generate coded design
    if python3 "/app/platform_src/design_generation/1_build_coded_design.py" "$chosen_dir"; then
        echo -e "Coded Design generated."
    else
        echo -e "Generating Coded Design encountered an error. Aborting further execution."
        break
    fi

    ############################################################################
    # generate design with real values

    if python3 "/app/platform_src/design_generation/2_generate_final_design.py" "$chosen_dir"; then
        echo -e "Real Values Design generated."
    else
        echo -e "Generating Real Values Design encountered an error. Aborting further execution."
        break
    fi

    ############################################################################
    # assign wells

    if python3 "/app/platform_src/design_generation/3_assign_wells.py" "$chosen_dir"; then
        echo -e "Wells were assigned to the design."
    else
        echo -e "Assigning wells encountered an error. Aborting further execution."
        break
    fi

    ############################################################################
    # generate heatmap
    if python3 "/app/platform_src/design_generation/conditions_heatmap.py" "$chosen_dir"; then
        echo -e "Design heatmap generated."
    else
        echo -e "Assigning wells encountered an error. Aborting further execution."
        break
    fi

    ############################################################################
    # add columns for data entry
    if python3 "/app/platform_src/design_generation/append_Y_cols.py" "$chosen_dir"; then
        echo -e "Appended Y columns for data entry."
    else
        echo -e "Assigning wells encountered an error. Aborting further execution."
        break
    fi
}

new_design() {

    # spacer
    echo -e "\n"

    local chosen_dir=$chosen_dir

        
    # Check if experimental designs have already been generated and ask if want to skip to analysis

    if [ -e "$chosen_dir/Experiment_Designs/design_real_values.csv" ]; then
        echo "An design has already been generated for this project. Would you like to regenerate the design? (y/n)"
        
        read user_input

        # Convert the input to lowercase for case-insensitive comparison
        user_input_lowercase=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')

        # IF NO - return
        if [[ "$user_input_lowercase" == "n" ]]; then
            
            echo -e "Returning to Project Menu.."
            echo -e "\n"
            break
        else
            generate_design
            
            cp "$chosen_dir/Experiment_Designs/design_real_values.csv" "$chosen_dir/Experiment_Designs/final_design.csv"
            chmod 777 $chosen_dir/Experiment_Designs/*

            echo -e "\n"
            echo -e "Design process complete."
            echo -e "The final_design.csv is located in /Experiment_Designs of the project folder."
            echo -e "Populate the design with your response variable data and place a copy in the "/Place Input Files Here" directory for upload."
            echo -e "\n"
        fi

    else
        generate_design
        cp "$chosen_dir/Experiment_Designs/design_real_values.csv" "$chosen_dir/Experiment_Designs/final_design.csv"
        chmod 777 $chosen_dir/Experiment_Designs/*

        echo -e "\n"
        echo -e "Design process complete."
        echo -e "The final_design.csv is located in /Experiment_Designs of the project folder."
        echo -e "Populate the design with your response variable data and place a copy in the "/Place Input Files Here" directory for upload."
        echo -e "\n"
    fi

}

