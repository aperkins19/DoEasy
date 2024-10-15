#!/bin/bash

source /app/platform_src/raw_data_preprocessing/create_tidydataset.sh
source /app/platform_src/raw_data_preprocessing/remove_outlier.sh
source /app/platform_src/raw_data_preprocessing/feature_extraction.sh
source /app/platform_src/raw_data_preprocessing/calibration.sh
source /app/platform_src/raw_data_preprocessing/plot_all_conditions.sh



preprocessing_menu() {

    echo -e "\n"
    echo -e "Trueasdfasdfasdfdf"
    
    # get variables
    local chosen_dir=$chosen_dir
    
    # preprocessing loop
    while true; do
        echo "Preprocessing:"
        echo "1. Create Tidy Dataset from rawdata and experiment params"
        echo "2. Plot All Conditions"
        echo "3. Remove outliers"
        echo "4. Calibration"
        echo "5. Feature Extraction"
        echo "6. Return to Project Selection"
        read -p "Enter choice [1-5]: " projectmenuchoice

        case $projectmenuchoice in
            1) 
                if create_tidydataset; then
                    echo "create_tidydataset complete."
                else
                    echo "create_tidydataset failed. plot_all_conditions will not execute."
                fi;;
            2)
                if plot_all_conditions; then
                    echo "Plot complete."
                else
                    echo "remove_outlier failed. plot_all_conditions will not execute."
                fi;;
            3)
                if remove_outlier; then
                    plot_all_conditions
                else
                    echo "remove_outlier failed. plot_all_conditions will not execute."
                fi;;
            4)
                if calibration; then
                    plot_all_conditions
                else
                    echo "calibration failed. plot_all_conditions will not execute."
                fi;;
            5)
                if feature_extraction; then
                    echo "Feature extraction complete."
                else
                    echo "feature_extraction failed. plot_all_conditions will not execute."
                fi;;
            6) echo "Returning"
                break # This exits the sub-menu loop
                ;;
            *) echo "Invalid choice, try again";;
        esac
    done
}
