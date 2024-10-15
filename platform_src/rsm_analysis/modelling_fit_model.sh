#! /bin/bash

fit_and_assess() {

    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # fit appropriate model
    if [ "$modeltype" == "deterministiclinearregression" ]; then

        # if model fit completes successfully execute assess
        if python3 "/app/platform_src/rsm_analysis/modelling_fit_model.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
            echo -e "Model Fit completed."

            if python3 "/app/platform_src/rsm_analysis/modelling_assess_model.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$modeltype"; then
                echo -e "Model Assessment completed."

            else
                echo -e "Model Assessment encountered an error. Aborting further execution."
            fi
        else
            echo -e "Model Fitting encountered an error. Aborting further execution."
        fi

    elif [ "$modeltype" == "bayesianregression" ]; then

        echo -e "Bayesian inference not yet implemented"

    else
        echo "unknown modeltype"
    fi 
}