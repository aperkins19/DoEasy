#! /bin/bash

find_predicted_max() {

    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # fit appropriate model
    if [ "$modeltype" == "deterministiclinearregression" ]; then


        # theoretical outcomes
        if python3 "/app/platform_src/rsm_analysis/modelling_find_predicted_max.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
            echo -e "Theoretical Max found."

        else
            echo -e "Model Theoretical Interegation encountered an error. Aborting further execution."
        fi



    elif [ "$modeltype" == "bayesianregression" ]; then

        echo -e "Bayesian inference not yet implemented"

    else
        echo "unknown modeltype"
    fi 
}