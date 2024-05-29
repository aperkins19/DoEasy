
# initialise_linear_model_json() {
#     # get array of variables from keys of dict
#     mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)
#     echo -e "Initialising first order model containing all variables but no interaction terms."

#     # Convert Bash array to a JSON array
#     json_array=$(printf '%s\n' "${Variables[@]}" | jq -R . | jq -s .)

#     # Use jq to build the JSON object with the array
#     model_config_init_json=$(jq -n --argjson Variables "$json_array" '{
#         "Degree": 1,
#         "model_terms": $Variables
#     }')

#     # Write the JSON object to a file
#     echo "$model_config_init_json" > "$chosen_dir/screening_model_config.json"
# }

# initialise_quadratic_model_json() {

#     # first delete any existing model
#     rm "$chosen_dir/screening_model_config.json"

#     # Get array of variables from keys of dict
#     mapfile -t Variables < <(jq -r '.Variables | keys[]' "$chosen_dir/design_parameters.json")
    
#     # initalise linear terms
#     # Convert Bash array to a JSON array
#     json_array=$(printf '%s\n' "${Variables[@]}" | jq -R . | jq -s .)

#     # Use jq to build the JSON object with the array
#     model_config_init_json=$(jq -n --argjson Variables "$json_array" '{
#         "Degree": 1,
#         "model_terms": $Variables
#     }')

#     # Write the JSON object to a file
#     echo "$model_config_init_json" > "$chosen_dir/screening_model_config.json"

#     ## add all interaction terms

#     # Initialize an array to store the pairs
#     local pairs=()
#     local i j pair

#     # Generate all unique pairs
#     for (( i = 0; i < ${#Variables[@]}; i++ )); do
#         for (( j = i + 1; j < ${#Variables[@]}; j++ )); do
#             pair="${Variables[i]}.${Variables[j]}"
#             pairs+=("$pair")
#         done
#     done

#     # append pairs
#     for pair in "${pairs[@]}"; do
#             # Read the current array from JSON, append the selected option, and update the JSON file
#             jq --arg option "$pair" '.model_terms += [$option]' "$chosen_dir/screening_model_config.json" > tmpfile && mv tmpfile "$chosen_dir/screening_model_config.json"
#     done

#     ## add all quadratic terms

#     # Initialize an array to store the squared variable strings
#     squared_vars=()
#     local var squared_var

#     # Generate squared variable strings
#     for var in "${Variables[@]}"; do
#         squared_var="${var}**2"
#         squared_vars+=("$squared_var")
#     done

#     # Optional: Print squared variable strings for verification
#     for squared_var in "${squared_vars[@]}"; do
#         # Read the current array from JSON, append the selected option, and update the JSON file
#         jq --arg option "$squared_var" '.model_terms += [$option]' "$chosen_dir/screening_model_config.json" > tmpfile && mv tmpfile "$chosen_dir/screening_model_config.json"

#     done



# }


# delete_interaction() {
#     local json_file="$chosen_dir/screening_model_config.json"
#     local key="model_terms"

#     # Read the array from the JSON file
#     mapfile -t array_elements < <(jq -r ".${key}[]" "$json_file")

#     echo "Select an element to delete:"
#     select element_to_remove in "${array_elements[@]}"; do
#         if [[ -n $element_to_remove ]]; then
#             # Use jq to delete the element from the array and update the file
#             jq --arg key "$key" --arg element "$element_to_remove" '
#               (.[$key] | index($element)) as $idx |
#               if $idx then .[$key][$idx] |= empty else . end' "$json_file" > tmpfile && mv tmpfile "$json_file"
#             echo "Element '$element_to_remove' has been removed."
#             break
#         else
#             echo "Invalid selection. Please try again."
#         fi
#     done
# }



# # Function to generate and select pair combinations
# add_interaction() {
#     # get array of variables from keys of dict
#     mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

#     select option in "${Variables[@]}"; do
#         if [ -n "$option" ]; then
#             echo "You selected: $option"
#             selected=$option
#             # Read the current array from JSON, append the selected option, and update the JSON file
#             jq --arg option "$selected" '.model_terms += [$option]' "$chosen_dir/screening_model_config.json" > tmpfile && mv tmpfile "$chosen_dir/screening_model_config.json"
        
#             break
#         else
#             echo "Please enter a valid option."
#         fi
#     done
# }

# add_quadratic_interaction() {
#     # get array of variables from keys of dict
#     mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

#     select option in "${Variables[@]}"; do
#         if [ -n "$option" ]; then
#             echo "You selected: $option"
#             selected="$option**2"
#             # Read the current array from JSON, append the selected option, and update the JSON file
#             jq --arg option "$selected" '.model_terms += [$option]' "$chosen_dir/screening_model_config.json" > tmpfile && mv tmpfile "$chosen_dir/screening_model_config.json"
        
#             break
#         else
#             echo "Please enter a valid option."
#         fi
#     done
# }

# # Function to generate and select pair combinations
# add_two_way_interaction() {
#     # Get array of variables from keys of dict
#     mapfile -t Variables < <(jq -r '.Variables | keys[]' "$chosen_dir/design_parameters.json")

#     # Initialize an array to store the pairs
#     local pairs=()
#     local i j pair

#     # Generate all unique pairs
#     for (( i = 0; i < ${#Variables[@]}; i++ )); do
#         for (( j = i + 1; j < ${#Variables[@]}; j++ )); do
#             pair="${Variables[i]}.${Variables[j]}"
#             pairs+=("$pair")
#         done
#     done


#     # Display a selectable list
#     PS3="Select a pair of strings: "
#     select option in "${pairs[@]}"; do
#         if [ -n "$option" ]; then
#             echo "You selected: $option"
#             selected=$option
#             # Read the current array from JSON, append the selected option, and update the JSON file
#             jq --arg option "$selected" '.model_terms += [$option]' "$chosen_dir/screening_model_config.json" > tmpfile && mv tmpfile "$chosen_dir/screening_model_config.json"
        

#             break
#         else
#             echo "Please enter a valid option."
#         fi
#     done
# }



# add_three_way_interaction() {
#         # get array of variables from keys of dict
#         mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)
#         local groups=()
#         local i j k group

#         # Generate all unique groups of three
#         for (( i = 0; i < ${#Variables[@]} - 2; i++ )); do
#             for (( j = i + 1; j < ${#Variables[@]} - 1; j++ )); do
#                 for (( k = j + 1; k < ${#Variables[@]}; k++ )); do
#                     group="${Variables[i]}.${Variables[j]}.${Variables[k]}"
#                     groups+=("$group")
#                 done
#             done
#         done

#         # Display a selectable list
#         PS3="Select a group of three strings: "
#         select option in "${groups[@]}"; do
#             if [ -n "$option" ]; then
#                 echo "You selected: $option"
#                 selected=$option
#                 # Read the current array from JSON, append the selected option, and update the JSON file
#                 jq --arg option "$selected" '.model_terms += [$option]' "$chosen_dir/screening_model_config.json" > tmpfile && mv tmpfile "$chosen_dir/screening_model_config.json"
        
#                 break
#             else
#                 echo "Please enter a valid option."
#             fi
#         done
#     }




# source /app/platform_src/screening_analysis/main_effects_plot.sh
# source /app/platform_src/screening_analysis/individual_effects_plot.sh

# screening_analysis_menu() {

#     # spacer
#     echo -e "\n"
 
#     # get array of variables from keys of dict
#     mapfile -t Variables < <(jq -r '.Variables | keys[]' $chosen_dir/design_parameters.json)

#     # Check if the file exists
#     if [ -f "$chosen_dir/screening_model_config.json" ]; then
#             # spacer
#     echo -e "\n"
#     echo -e "Conduct Analysis of the $designtype design."
#     echo -e "\n"
#     # Sub-menu loop
#     while true; do

#         echo -e "\n"
#         # retrieve and display interaction terms
#         mapfile -t Terms < <(jq -r '.model_terms[]' "$chosen_dir/screening_model_config.json")
#         echo -e "Current Model Terms:"
#         # Iterate over the array and print each variable
#         for term in "${Terms[@]}"; do
#             echo "- $term"
#         done

#         echo -e "\n"
#         echo -e "What would you like to do:"
#         echo -e "\n"
#         echo -e "Use a standard model:"
#         echo -e "1. Linear Model"
#         echo -e "2. Fully-Featured Quadratic Model"
#         echo -e "\n"
#         echo -e "Custom Modelling:"
#         echo -e "3. Add Single Interaction"
#         echo -e "4. Add Two-Way Interaction"
#         echo -e "5. Add Three-Way Interaction"
#         echo -e "6. Add Quadratic Interaction"
#         echo -e "7. Delete Interaction"
#         echo -e "\n"
#         echo -e "Analysis:"
#         echo -e "8. Main Effects Analysis"
#         echo -e "9. Individual Effects Analysis"
#         echo -e "10. Return to Project Main Menu"
#         read -p "Enter choice [1-10]: " screeninganalysismenuchoice

#         case $screeninganalysismenuchoice in
#             1)  initialise_linear_model_json
#                 ;;
#             2)  initialise_quadratic_model_json
#                 ;;
                
#             3)  add_interaction
#                 echo -e "\n"
#                 ;;
#             4) add_two_way_interaction
#                 echo -e "\n"
#                 ;;

#             5) add_three_way_interaction
#                 echo -e "\n"
#                 ;;
            
#             6)  add_quadratic_interaction
#                 echo -e "\n"
#                 ;;

#             7) delete_interaction
#                 echo -e "\n"
#                 ;; 

#             8) main_effects_plot
#                 echo -e "\n"
#                 ;;

#             9) individual_effects_plot
#                 echo -e "\n"
#                 ;;

#             10)  
#                 echo "Returning"
#                 break # This exits the sub-menu loop
#                 ;;
#             *) 
#                 echo -e "\n"
#                 echo "Invalid choice, try again"
#                 echo -e "\n"
#                 ;;
#         esac
#     done

#     else
#         initialise_linear_model_json
#     fi
# }
   