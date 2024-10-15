#!/bin/bash

source /app/platform_src/configure_experiment_parameters.sh
source /app/platform_src//design_generation/generate_design.sh
source /app/platform_src/upload_raw_data_csv.sh
source /app/platform_src/preprocessing_menu.sh
source /app/platform_src/response_variables_top_menu.sh
source /app/platform_src/screening_analysis/screening_analysis_menu.sh


project_menu() {
    # spacer
    echo -e "\n"
    
    # Sub-menu loop
    while true; do

        # get params
        option_counter=4
        preprocessing_option_available=true
        analysis_menu_option_available=false
        # get design type
        designtype=$(jq -r '.Design_Type' $chosen_dir/design_parameters.json)

        
        # print menu
        # Always
        echo "What would you like to do:"
        echo -e "\n"
        echo "1. Configure Experiment Parameters"
        echo "2. Generate Design"
        echo "3. Upload .CSV file of Raw Data."

        # if raw data but no tidy_dataset: preprocessing available
        if [ -f "$chosen_dir/Datasets/RawData/input_raw_data_dataset.csv" ]; then
            if [ ! -f "$chosen_dir/Datasets/tidy_dataset.csv" ]; then

                echo -e "\n"
                echo "A Tidy Dataset is not present, either re-upload raw data in the tidy format or:"
                echo "$option_counter. Conduct Preprocessing"

                # Increment option_counter by 1
                ((option_counter++))
                preprocessing_option_available=true
            fi
        fi

        ######## if tidy_dataset: Response Variables Menu
        if  [[ -f "$chosen_dir/Datasets/tidy_dataset.csv" ]]; then

            echo -e "\n"
            echo "To conduct Preprocessing progress to: "
            echo -e "$option_counter. Conduct Preprocessing"


            ((option_counter++))
            preprocessing_option_available=true

            echo -e "\n"
            echo "To conduct analysis progress to: "

            # check with analysis pipeline to display based on design type
            if [[ "$designtype" == "Plackett_Burman" ]] || [[ "$designtype" == "Simple_Screening_Design" ]]; then
                echo "$option_counter. Screening Analysis Menu"


            elif [[ "$designtype" == "CCD" ]] || [[ "$designtype" == "Full_Factorial" ]]; then
                echo "$option_counter. Response Variables Menu"
            fi

            ((option_counter++))
            analysis_menu_option_available=true
        fi

        echo -e "\n"
        echo "$option_counter. Return to Project Selection"
        read -p "Enter choice [1-$option_counter]: " projectmenuchoice




        case $projectmenuchoice in
            1) configure_experiment_parameters;;
            2) new_design;;
            3) upload_raw_data_csv;;

            4)
                if [ "$preprocessing_option_available" = true ] ; then
                    preprocessing_menu
                # if raw data but no tidy_dataset
                elif [ "$preprocessing_option_available" = false ]; then
                    echo -e "Please upload CSV to conduct preprocessing"
                    echo "Returning"
                    break # This exits the sub-menu loop
                fi
                ;;
            5)
                if [ "$analysis_menu_option_available" = true ] ; then
                    # check with analysis pipeline to display based on design type
                    if [[ "$designtype" == "Plackett_Burman" ]] || [[ "$designtype" == "Simple_Screening_Design" ]]; then
                        screening_analysis_menu

                    elif [[ "$designtype" == "CCD" ]] || [[ "$designtype" == "Full_Factorial" ]]; then
                        response_variables_top_menu
                    fi
                elif [ "$analysis_menu_option_available" = false ]; then
                    echo -e "Please upload CSV to conduct preprocessing"
                     echo "Returning"
                     break # This exits the sub-menu loop

                fi
                ;;


            6)
                break
#                 if [ "$preprocessing_option_available" = false ] && [ "$analysis_menu_option_available" = true ]; then
#                     echo "Returning"
#                     break # This exits the sub-menu loop
#
#                 elif [ "$preprocessing_option_available" = true ] && [ "$analysis_menu_option_available" = false ]; then
#                     echo "Returning"
#                     break # This exits the sub-menu loop
#                 fi
                ;;

            *) echo "Invalid choice, try again";;
        esac
    done
}

