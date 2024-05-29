#! /bin/bash

term_significance() {

    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # fit appropriate model
    if [ "$modeltype" == "deterministiclinearregression" ]; then


        # theoretical outcomes
        if python3 "/app/platform_src/rsm_analysis/modelling_term_significance.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
            echo -e "Model Term Significance Analysis Complete."

        else
            echo -e "Model Term Significance Analysis encountered an error. Aborting further execution."
        fi



    elif [ "$modeltype" == "bayesianregression" ]; then

        echo -e "Bayesian inference not yet implemented"

    else
        echo "unknown modeltype"
    fi 
}