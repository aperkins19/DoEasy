#!/bin/bash

source /app/platform_src/rsm_analysis/error_and_noise_analysis.sh
source /app/platform_src/rsm_analysis/plusminus_to_ref_bar.sh
source /app/platform_src/rsm_analysis/plotY_ranked_by_X.sh
source /app/platform_src/rsm_analysis/modelling_menu.sh

rsm_analysis_menu() {
    # spacer
    echo -e "\n"
    echo -e "Analysis is performed with respect to a specfic response variables."
    echo -e "\n"
    # Sub-menu loop
    while true; do
        echo -e "What would you like to do:"
        echo -e "1. Error & Noise Analysis"
        echo -e "2. Relative Performance to Reference Condition"
        echo -e "3. Plot Data Ranked by Variable"
        echo -e "4. Modelling"
        echo -e "5. Return to Project Main Menu"
        read -p "Enter choice [1-5]: " rsmanalysismenuchoice

        case $rsmanalysismenuchoice in
            1) error_and_noise_analysis;;
            2) plusminus_to_ref_bar;;
            3) plotY_ranked_by_X;;
            4) modelling_menu;;
            5)  
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
}

