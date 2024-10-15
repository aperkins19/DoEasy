#! /bin/bash

specific_slices() {

    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # reset the dir
    rm -f "${select_model_dir}/individual_slices/"*

    # fit appropriate model
    if [ "$modeltype" == "deterministiclinearregression" ]; then


        # theoretical outcomes
        if python3 "/app/platform_src/rsm_analysis/modelling_specific_slices.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
            echo -e "Model Specific Slices Complete."

        else
            echo -e "Model Term Significance Analysis encountered an error. Aborting further execution."
        fi



    elif [ "$modeltype" == "bayesianregression" ]; then

        echo -e "Bayesian inference not yet implemented"

    else
        echo "unknown modeltype"
    fi 
}
