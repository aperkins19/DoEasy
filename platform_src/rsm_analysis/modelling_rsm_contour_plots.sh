#! /bin/bash

rsm_contour_plots() {

    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # fit appropriate model
    if [ "$modeltype" == "deterministiclinearregression" ]; then

        if python3 "/app/platform_src/rsm_analysis/modelling_rsm_contour_plots.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
            echo -e "Contour Plots completed."

        else
            echo -e "Contour Plots encountered an error. Aborting further execution."
        fi

    elif [ "$modeltype" == "bayesianregression" ]; then

        echo -e "Bayesian inference not yet implemented"

    else
        echo "unknown modeltype"
    fi 
}
