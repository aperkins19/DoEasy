#!/bin/bash

# Define functions for different actions
error_and_noise_analysis() {

    # import
    chosen_dir=$chosen_dir
    feature_to_model=$feature_to_model
    output="Output/"
    explore="/explore"
    savepath="$chosen_dir$output$feature_to_model$explore"

    # spacer
    echo -e "\n"

    echo -e "Error and noise"
    echo -e "\n"

    if python3 "/app/platform_src/rsm_analysis/error_and_noise.py" "$chosen_dir" "$savepath" "$feature_to_model"; then
        echo -e "Noise analysis complete.."
        echo -e "\n"
    else
        echo -e "Noise analysis encountered an error. Aborting further execution."
        echo -e "\n"
    fi
    #echo -e "Please configure the experimental parameters before generating your design."
    #echo -e "\n"

}
