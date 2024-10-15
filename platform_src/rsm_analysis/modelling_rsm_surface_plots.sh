#! /bin/bash

rsm_surface_plots() {

    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # fit appropriate model
    if [ "$modeltype" == "deterministiclinearregression" ]; then

        if python3 "/app/platform_src/rsm_analysis/modelling_rsm_surface_plots.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
            echo -e "Surface Plots completed."

        else
            echo -e "RSM encountered an error. Aborting further execution."
        fi

    elif [ "$modeltype" == "bayesianregression" ]; then

        echo -e "Bayesian inference not yet implemented"

    else
        echo "unknown modeltype"
    fi 
}
