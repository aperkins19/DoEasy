#! /bin/bash

individual_effects_plot() {

    # Use the variables
    # model dir = $select_model_dir"

    # fit appropriate model
    if [ "$designtype" == "Plackett_Burman" ]; then

        if python3 "/app/platform_src/screening_analysis/individual_effects_plot.py" "$chosen_dir"; then
            echo -e "Individual Effects Analysis Complete."
        else
            echo -e "Individual Effects Analysis encountered an error. Aborting further execution."
        fi

    elif [ "$designtype" == "Simple_Screening_Design" ]; then
        
        echo "Simple screening design analysis yet to be implemented"

    else
        echo "unknown modeltype"
    fi 
}