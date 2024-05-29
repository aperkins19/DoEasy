#!/bin/bash

# Define functions for different actions
plusminus_to_ref_bar() {

    # import
    chosen_dir=$chosen_dir
    feature_to_model=$feature_to_model
    output="Output/"
    explore="/explore"
    savepath="$chosen_dir$output$feature_to_model$explore"

    # spacer
    echo -e "\n"

    echo -e "Relative Performance to Reference Condition"
    echo -e "\n"

    if python3 "/app/platform_src/rsm_analysis/plusminus_to_ref_bar.py" "$chosen_dir" "$savepath" "$feature_to_model"; then
        echo -e "Relative Performance to Reference Condition plotted in /explore."
        echo -e "\n"
    else
        echo -e "Relative Performance to Reference Condition encountered an error. Aborting further execution."
        echo -e "\n"
    fi
    #echo -e "Please configure the experimental parameters before generating your design."
    #echo -e "\n"

}





########### make plus and minus from reference.