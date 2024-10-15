#!/bin/bash

# Function to generate and select pair combinations
rsm_add_intercept() {
    # get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

    selected="Intercept"
    # Read the current array from JSON, append the selected option, and update the JSON file
    jq --arg option "$selected" '.model_terms += [$option]' "${select_model_dir}/model_params.json" > tmpfile && mv tmpfile "${select_model_dir}/model_params.json"


}


rsm_initialise_linear_model_json() {


    echo -e "Initialising first order model containing all variables but no interaction terms."

    # get array of  variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys_unsorted[]' "$chosen_dir/design_parameters.json")



    # Convert Bash array to a JSON array
    json_array=$(printf '%s\n' "${Variables[@]}" | jq -R . | jq -s .)

    echo -e $json_array


    # Initialize the JSON object with "Intercept" as the first model term
    model_config_init_json=$(jq -n '{
        "model_terms": ["Intercept"]
    }')

    # Use jq to append the other variables to the "model_terms" array
    model_config_final_json=$(echo "$model_config_init_json" | jq --argjson Variables "$json_array" '.model_terms += $Variables')

    # Write the final JSON object to a file
    echo "$model_config_final_json" > "${select_model_dir}/model_params.json"
}


rsm_initialise_quadratic_model_json() {
    # first delete any existing model
    rm -f "${select_model_dir}/model_params.json"

    # Get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys_unsorted[]' "$chosen_dir/design_parameters.json")
    
    # Convert Bash array to a JSON array for linear terms
    json_array=$(printf '%s\n' "${Variables[@]}" | jq -R . | jq -s .)

    # Initialize the JSON object with "Intercept" as the first model term
    model_config_init_json=$(jq -n '{
        "model_terms": ["Intercept"]
    }')

    # Use jq to append the other variables to the "model_terms" array
    model_config_final_json=$(echo "$model_config_init_json" | jq --argjson Variables "$json_array" '.model_terms += $Variables')


    ## Prepare all interaction terms, excluding interactions with "Intercept"
    local interaction_terms=()
    local quadratic_terms=()

    # Generate all unique pairs for interaction terms
    for (( i = 0; i < ${#Variables[@]}; i++ )); do
        for (( j = i + 1; j < ${#Variables[@]}; j++ )); do
            interaction_terms+=("${Variables[i]}.${Variables[j]}")
        done
    done

    # Generate quadratic terms for each variable
    for var in "${Variables[@]}"; do
        quadratic_terms+=("${var}**2")
    done

    # Combine interaction and quadratic terms
    all_terms=("${interaction_terms[@]}" "${quadratic_terms[@]}")

    # Convert combined terms to JSON array
    json_terms=$(printf '%s\n' "${all_terms[@]}" | jq -R . | jq -s .)

    # Append the interaction and quadratic terms to the model_terms array
    model_config_final_json=$(echo "$model_config_final_json" | jq --argjson Terms "$json_terms" '.model_terms += $Terms')

    # Write the final JSON object to a file
    echo "$model_config_final_json" > "${select_model_dir}/model_params.json"
}


rsm_initialise_cubic_model_json() {
    # first delete any existing model
    rm -f "${select_model_dir}/model_params.json"

    # Get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys_unsorted[]' "$chosen_dir/design_parameters.json")
    
    # Convert Bash array to a JSON array for linear terms
    json_array=$(printf '%s\n' "${Variables[@]}" | jq -R . | jq -s .)

    # Initialize the JSON object with "Intercept" as the first model term
    model_config_init_json=$(jq -n '{
        "model_terms": ["Intercept"]
    }')

    # Use jq to append the other variables to the "model_terms" array
    model_config_final_json=$(echo "$model_config_init_json" | jq --argjson Variables "$json_array" '.model_terms += $Variables')


    ## Prepare all interaction terms, excluding interactions with "Intercept"
    local interaction_terms=()
    local quadratic_terms=()

    # Generate all unique pairs for interaction terms
    for (( i = 0; i < ${#Variables[@]}; i++ )); do
        for (( j = i + 1; j < ${#Variables[@]}; j++ )); do
            interaction_terms+=("${Variables[i]}.${Variables[j]}")
        done
    done

    # Generate quadratic terms for each variable
    for var in "${Variables[@]}"; do
        quadratic_terms+=("${var}**2")
    done

    # Generate all unique three-way pairs for interaction terms
    for (( i = 0; i < ${#Variables[@]}; i++ )); do
        for (( j = i + 1; j < ${#Variables[@]}; j++ )); do
            for (( k = j + 1; k < ${#Variables[@]}; k++ )); do
                three_way_interaction="${Variables[i]}.${Variables[j]}.${Variables[k]}"
                interaction_terms+=("$three_way_interaction")
            done
        done
    done

    # Generate cubic terms for each variable
    for var in "${Variables[@]}"; do
        cubic_terms+=("${var}**3")
    done


    # Combine interaction and quadratic terms
    all_terms=("${interaction_terms[@]}" "${quadratic_terms[@]}" "${cubic_terms[@]}")

    # Convert combined terms to JSON array
    json_terms=$(printf '%s\n' "${all_terms[@]}" | jq -R . | jq -s .)

    # Append the interaction and quadratic terms to the model_terms array
    model_config_final_json=$(echo "$model_config_final_json" | jq --argjson Terms "$json_terms" '.model_terms += $Terms')

    # Write the final JSON object to a file
    echo "$model_config_final_json" > "${select_model_dir}/model_params.json"
}



rsm_delete_interaction() {
    local json_file="${select_model_dir}/model_params.json"
    local key="model_terms"

    # Read the array from the JSON file
    mapfile -t array_elements < <(jq -r ".${key}[]" "$json_file")

    echo "Select an element to delete:"
    select element_to_remove in "${array_elements[@]}"; do
        if [[ -n $element_to_remove ]]; then
            # Use jq to delete the element from the array and update the file
            jq --arg key "$key" --arg element "$element_to_remove" '
              (.[$key] | index($element)) as $idx |
              if $idx then .[$key][$idx] |= empty else . end' "$json_file" > tmpfile && mv tmpfile "$json_file"
            echo "Element '$element_to_remove' has been removed."
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done
}


# Function to generate and select pair combinations
rsm_add_interaction() {
    # get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

    select option in "${Variables[@]}"; do
        if [ -n "$option" ]; then
            echo "You selected: $option"
            selected=$option
            # Read the current array from JSON, append the selected option, and update the JSON file
            jq --arg option "$selected" '.model_terms += [$option]' "${select_model_dir}/model_params.json" > tmpfile && mv tmpfile "${select_model_dir}/model_params.json"
        
            break
        else
            echo "Please enter a valid option."
        fi
    done
}

rsm_add_quadratic_term() {
    # get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

    select option in "${Variables[@]}"; do
        if [ -n "$option" ]; then
            echo "You selected: $option"
            selected="$option**2"
            # Read the current array from JSON, append the selected option, and update the JSON file
            jq --arg option "$selected" '.model_terms += [$option]' "${select_model_dir}/model_params.json" > tmpfile && mv tmpfile "${select_model_dir}/model_params.json"
        
            break
        else
            echo "Please enter a valid option."
        fi
    done
}

rsm_add_cubic_term() {
    # get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

    select option in "${Variables[@]}"; do
        if [ -n "$option" ]; then
            echo "You selected: $option"
            selected="$option**3"
            # Read the current array from JSON, append the selected option, and update the JSON file
            jq --arg option "$selected" '.model_terms += [$option]' "${select_model_dir}/model_params.json" > tmpfile && mv tmpfile "${select_model_dir}/model_params.json"

            break
        else
            echo "Please enter a valid option."
        fi
    done
}

# Function to generate and select pair combinations
rsm_add_two_way_interaction() {
    # Get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys_unsorted[]' "$chosen_dir/design_parameters.json")

    # Initialize an array to store the pairs
    local pairs=()
    local i j pair

    # Generate all unique pairs
    for (( i = 0; i < ${#Variables[@]}; i++ )); do
        for (( j = i + 1; j < ${#Variables[@]}; j++ )); do
            pair="${Variables[i]}.${Variables[j]}"
            pairs+=("$pair")
        done
    done


    # Display a selectable list
    PS3="Select a pair of strings: "
    select option in "${pairs[@]}"; do
        if [ -n "$option" ]; then
            echo "You selected: $option"
            selected=$option
            # Read the current array from JSON, append the selected option, and update the JSON file
            jq --arg option "$selected" '.model_terms += [$option]' "${select_model_dir}/model_params.json" > tmpfile && mv tmpfile "${select_model_dir}/model_params.json"
        

            break
        else
            echo "Please enter a valid option."
        fi
    done
}



rsm_add_three_way_interaction() {
        # get array of variables from keys of dict
        mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)
        local groups=()
        local i j k group

        # Generate all unique groups of three
        for (( i = 0; i < ${#Variables[@]} - 2; i++ )); do
            for (( j = i + 1; j < ${#Variables[@]} - 1; j++ )); do
                for (( k = j + 1; k < ${#Variables[@]}; k++ )); do
                    group="${Variables[i]}.${Variables[j]}.${Variables[k]}"
                    groups+=("$group")
                done
            done
        done

        # Display a selectable list
        PS3="Select a group of three strings: "
        select option in "${groups[@]}"; do
            if [ -n "$option" ]; then
                echo "You selected: $option"
                selected=$option
                # Read the current array from JSON, append the selected option, and update the JSON file
                jq --arg option "$selected" '.model_terms += [$option]' "${select_model_dir}/model_params.json" > tmpfile && mv tmpfile "${select_model_dir}/model_params.json"
        
                break
            else
                echo "Please enter a valid option."
            fi
        done
    }


refresh_input_variables_and_unique_levels() {

    #levels
    if python3 "/app/platform_src/rsm_analysis/sub_scripts/utils/get_unique_levels.py" "$chosen_dir" "$select_model_dir"; then
        echo -e "Unique levels identification completed."

    else
        echo -e "Unique levels identification encountered an error. Aborting further execution."
    fi

}



source /app/platform_src/rsm_analysis/modelling_fit_model.sh
source /app/platform_src/rsm_analysis/modelling_rsm_contour_plots.sh
source /app/platform_src/rsm_analysis/modelling_rsm_surface_plots.sh
source /app/platform_src/rsm_analysis/modelling_find_predicted_max.sh
source /app/platform_src/rsm_analysis/modelling_term_significance.sh
# source /app/platform_src/rsm_analysis/modelling_generate_model_stats_and_plots.sh
source /app/platform_src/rsm_analysis/modelling_upload_prediction_csv.sh
source /app/platform_src/rsm_analysis/modelling_predict_csv.sh
source /app/platform_src/rsm_analysis/modelling_reset_predict_dir.sh
source /app/platform_src/rsm_analysis/modelling_specific_slices.sh
 



specific_model_menu() {

    # spacer
    echo -e "\n"

 
    # get array of variables from keys of dict
    mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

    echo -e "${select_model_dir}"

    # Check if the file exists
    if [ -f "${select_model_dir}/model_params.json" ]; then
        
        # spacer
        echo -e "\n"
        echo -e "Conduct Analysis of the $designtype design."
        echo -e "\n"
        # Sub-menu loop
        while true; do

            echo -e "\n"
            # retrieve and display interaction terms
            mapfile -t Terms < <(jq -r '.model_terms[]' "${select_model_dir}/model_params.json")
            echo -e "Current Model Terms:"
            # Iterate over the array and print each variable
            for term in "${Terms[@]}"; do
                echo "- $term"
            done

            echo -e "\n"
            echo -e "${select_model_dir}"
            echo -e "What would you like to do:"
            echo -e "\n"
            echo -e "Use a standard model:"
            echo -e "1. Linear Model"
            echo -e "2. Saturated Quadratic Model"
            echo -e "3. Saturated Cubic Model"
            echo -e "\n"
            echo -e "Model Editing:"
            echo -e "4. Add Intercept"
            echo -e "5. Add Linear Term"
            echo -e "6. Add Two-Way Interaction"
            echo -e "7. Add Three-Way Interaction"
            echo -e "8. Add Quadratic Term"
            echo -e "9. Add Cubic Term"
            echo -e "10. Delete Term"
            echo -e "\n"
            echo -e "Model Fitting & Response Surface Methodology:"
            echo -e "11. Fit"
            echo -e "12. Model Term Signficance Analysis"
            echo -e "13. Generate Contour Plots"
            echo -e "14. Generate Surface Plots"
            echo -e "15. Slice Plots"
            echo -e "16. Edit Model Config"
            echo -e "\n"
            echo -e "Predict"
            echo -e "17. Find the Predicted Optimium."
            echo -e "18. Upload .CSV file of Input Values for Prediction."
            echo -e "19. Predict."
            echo -e "20. Delete prediction input data, plots and output data and reset."
            echo -e "21. Return to Model Selection"
            read -p "Enter choice [1-21]: " screeninganalysismenuchoice

            case $screeninganalysismenuchoice in

                1)  rsm_initialise_linear_model_json
                    ;;

                2)  rsm_initialise_quadratic_model_json
                    ;;

                3)  rsm_initialise_cubic_model_json
                    ;;
                
                4)  rsm_add_intercept
                    echo -e "\n"
                    ;;

                5)  rsm_add_interaction
                    echo -e "\n"
                    ;;
                6) rsm_add_two_way_interaction
                    echo -e "\n"
                    ;;

                7) rsm_add_three_way_interaction
                    echo -e "\n"
                    ;;
                
                8)  rsm_add_quadratic_term
                    echo -e "\n"
                    ;;

                9)  rsm_add_cubic_term
                    echo -e "\n"
                    ;;

                10) rsm_delete_interaction
                    echo -e "\n"
                    ;; 

                11) fit_and_assess
                    refresh_input_variables_and_unique_levels
                    echo -e "\n"
                    ;;

                12) # check if fitted model exists
                    if [ -e "${select_model_dir}/fitted_model.pkl" ]; then
                        term_significance
                    else
                        echo -e "\n"
                        echo "Model has not yet been fitted. A model must be fitted before model terms can be analysed."
                    fi 
                    echo -e "\n"
                    ;;

                13) # check if fitted model exists
                    if [ -e "${select_model_dir}/fitted_model.pkl" ]; then
                        rsm_contour_plots
                    else
                        echo -e "\n"
                        echo "Model has not yet been fitted. A model must be fitted before plots of fit can be produced."
                    fi
                    ;;

                14) # check if fitted model exists
                    if [ -e "${select_model_dir}/fitted_model.pkl" ]; then
                        rsm_surface_plots
                    else
                        echo -e "\n"
                        echo "Model has not yet been fitted. A model must be fitted before plots of fit can be produced."
                    fi
                    ;;

                15) # check if fitted model exists
                    if [ -e "${select_model_dir}/fitted_model.pkl" ]; then
                        specific_slices
                    else
                        echo -e "\n"
                        echo "Model has not yet been fitted. A model must be fitted before plots of fit can be produced."
                    fi
                    ;;

                ## edit model config
                16)     # Check if the file exists
                        local model_config_file="model_config.json"
                        local model_config_path="$select_model_dir$model_config_file"
                        if [[ -f "$model_config_path" ]]; then
                            # Open the file in a text editor, e.g., nano
                            nano "$model_config_path"
                        else
                            echo -e "\n"
                            echo "JSON file not found: $model_config_path"
                        fi
                    echo -e "\n"
                    ;;



                17) # check if fitted model exists
                    if [ -e "${select_model_dir}/fitted_model.pkl" ]; then
                        find_predicted_max
                    else
                        echo -e "\n"
                        echo "Model has not yet been fitted. A model must be fitted before an optimum can be found."
                    fi;;

                18) upload_prediction_csv
                    ;;

                19) # check if fitted model exists
                    if [ -e "${select_model_dir}/fitted_model.pkl" ]; then
                        # check that file is uploaded
                        if [ -e "${select_model_dir}/prediction/input_prediction_dataset.csv" ]; then
                            predict_csv
                        else
                            echo -e "\n"
                            echo "An csv of conditions to be predicted has not been successfully uploaded. Please refer to the documentation to resolve."
                        fi
                    else
                        echo -e "\n"
                        echo "Model has not yet been fitted. A model must be fitted before a prediction can take place."
                    fi
                    ;;

                20) reset_predict_dir;;

                21)
                    echo "Returning"
                    break # This exits the sub-menu loop
                    ;;
                *) 
                    echo -e "\n"
                    echo "Invalid choice, try again"
                    echo -e "\n"
                    ;;
            esac
        done

    else
        rsm_initialise_linear_model_json
    fi
}
   

