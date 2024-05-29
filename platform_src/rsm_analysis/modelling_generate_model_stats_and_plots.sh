#! /bin/bash

generate_model_stats_and_plots() {

    # get model parameters as bash variables
    # get modeltype
    modeltype=$(jq -r '.Model_Type' $select_model_dir/model_config.json)

    # Use the variables
    echo "modeltype is $modeltype"
    # model dir = $select_model_dir"

    # predict appropriate model

    if python3 "/app/platform_src/rsm_analysis/modelling_generate_model_stats_and_plots.py" "$chosen_dir" "$select_model_dir" "$feature_to_model" "$select_model_name" "$modeltype"; then
        :
    else
        echo -e "Encountered an error. Aborting further execution."
    fi

}