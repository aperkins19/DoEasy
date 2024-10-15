#!/bin/bash

# Define functions for different actions
plotY_ranked_by_X() {

    # import
    chosen_dir=$chosen_dir
    feature_to_model=$feature_to_model
    output="Output/"
    explore="/explore"
    savepath="$chosen_dir$output$feature_to_model$explore"

    # spacer
    echo -e "\n"

    echo -e "Ranking $feature_to_model by every combination of input variables."
    echo -e "\n"

    if python3 "/app/platform_src/rsm_analysis/plotY_ranked_by_X.py" "$chosen_dir" "$savepath" "$feature_to_model"; then
        echo -e "Ranking complete, plots produced.."
        echo -e "\n"
    else
        echo -e "Ranking encountered an error. Aborting further execution."
        echo -e "\n"
    fi

}
