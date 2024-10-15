#!/bin/bash


get_input_variables_and_unique_levels() {

    #levels
    if python3 "/app/platform_src/rsm_analysis/sub_scripts/utils/get_unique_levels.py" "$chosen_dir" "$newmodeldir"; then
        echo -e "Unique levels identification completed."

    else
        echo -e "Unique levels identification encountered an error. Aborting further execution."
    fi

}



initialise_model_config_json(){


    # JSON file path
    model_config_json="/model_config.json"
    model_config_json_dir="$newmodeldir$model_config_json"
    # Start the JSON object
    echo "{" > $model_config_json_dir
    # Add key-value pairs
    ## model metadata
    echo "  \"Model_Name\": \"$modelname\"," >> $model_config_json_dir
    
    # model specific params
    if [ "$modeltype" == "deterministiclinearregression" ]; then
        # Start the JSON object
        echo "{" > $model_config_json_dir
        # Add key-value pairs
        ## model metadata
        echo "  \"Model_Name\": \"$modelname\"," >> $model_config_json_dir
        echo "  \"Model_Type\": \"$modeltype\"," >> $model_config_json_dir


        # Add plot_args dictionary
        echo "  \"plot_args\": {" >> $model_config_json_dir
        echo "    \"Surface\": {" >> $model_config_json_dir
        echo "      \"GIF\": true," >> $model_config_json_dir
        echo "      \"fps\": 0.5," >> $model_config_json_dir
        echo "      \"view_normal_elevation\": 15," >> $model_config_json_dir
        echo "      \"view_normal_azimuth\": 35," >> $model_config_json_dir
        echo "      \"plot_zero_surface\": true," >> $model_config_json_dir
        echo "      \"view_zero_surface_elevation\": 10," >> $model_config_json_dir
        echo "      \"view_zero_surface_azimuth\": 35," >> $model_config_json_dir
        echo "      \"plot_ground_truth\": false," >> $model_config_json_dir
        echo "      \"Fig_Size_x\": 10," >> $model_config_json_dir
        echo "      \"Fig_Size_y\": 10," >> $model_config_json_dir
        echo "      \"Suptitle_font_size\": 10," >> $model_config_json_dir
        echo "      \"Plot_title_font_size\": 10," >> $model_config_json_dir
        echo "      \"Axis_label_font_size\": 10" >> $model_config_json_dir
        echo "    }," >> $model_config_json_dir
        echo "    \"Contour\": {" >> $model_config_json_dir
        echo "      \"GIF\": true," >> $model_config_json_dir
        echo "      \"fps\": 0.5," >> $model_config_json_dir
        echo "      \"plot_ground_truth\": false," >> $model_config_json_dir
        echo "      \"Grid\": true," >> $model_config_json_dir
        echo "      \"Grid_colour\": \"grey\"," >> $model_config_json_dir
        echo "      \"Grid_linestyle\": \"--\"," >> $model_config_json_dir
        echo "      \"Grid_linewidth\": 0.5," >> $model_config_json_dir
        echo "      \"Fig_Size_x\": 10," >> $model_config_json_dir
        echo "      \"Fig_Size_y\": 10," >> $model_config_json_dir
        echo "      \"Ticks_use_levels\": true," >> $model_config_json_dir
        echo "      \"Ticks_label_size\": 40," >> $model_config_json_dir
        echo "      \"Suptitle_font_size\": 40," >> $model_config_json_dir
        echo "      \"Plot_title_font_size\": 40," >> $model_config_json_dir
        echo "      \"Axis_label_font_size\": 40" >> $model_config_json_dir
        echo "    }" >> $model_config_json_dir
        echo "  }," >> $model_config_json_dir

        echo "  \"Simulated_Annealing_Params\": {" >> $model_config_json_dir
        echo "  \"Temperature\": 5," >> $model_config_json_dir
        echo "  \"Cooling_Schedule\": 0.95," >> $model_config_json_dir
        echo "  \"N_Proposals\": 100," >> $model_config_json_dir
        echo "  \"particles\": 12000," >> $model_config_json_dir
        echo "  \"Deep_Search\": 0," >> $model_config_json_dir
        echo "  \"replicates\": 3," >> $model_config_json_dir
        echo "  \"decimal_point_threshold\": 1" >> $model_config_json_dir
        echo "}" >> $model_config_json_dir
        echo "}" >> $model_config_json_dir

    else
        echo "unknown modeltype"
    fi

    # get unique levels


}



initialise_model_directory() {

    # initialise directories
    # ModelsDirectory
    output="Output/"
    models="/models"
    modelsdirectory="$chosen_dir$output$feature_to_model$models"
    


    ### Choose model type

    while true; do

        echo -e "\n"
        echo -e "Select a model type:"
        echo -e "1) Determinisic Linear Regression"
        echo -e "2) Bayesian Regression (TBI)"
        echo -e "\n"
        echo -e "3) Back"

        read -p "Enter your choice [1-3]: " choice

        case $choice in
            1)
                echo "You selected Determinisic Linear Regression"
                modeltype="deterministiclinearregression"
                break
                ;;

            2)
                echo "You selected Bayesian Regression"
                modeltype="bayesianregression"
                break
                ;;
            3)
                echo "Exiting..."
                break
                ;;
            *)
                echo "Invalid choice."
                ;;
        esac
    done

    echo -e "\n"
    

    # model name
    while true; do
        read -p "Name the model: " modelname
        # check is name acceptible
        if [ -d "$modelsdirectory/$modelname" ]; then
            echo "Warning! A model with this name already exists: $modelname, please choose a different name.."
        else
            # replace all spaces with underscores and add slash at beginning
            modelname="${modelname// /_}"
            break
        fi
    done

    ## initialise the model directory
    newmodeldir="${modelsdirectory}/${modelname}/"
    mkdir -p "$newmodeldir"
    cp -r /app/platform_src/initialisation/Templates/Model_Template/* "$newmodeldir"
    ## initialise model config
    initialise_model_config_json

    # initialise plot directories
    contour="/contour_plots"
    contourdirectory="$newmodeldir$contour"
    mkdir -p "$contourdirectory"
    surface="/surface_plots"
    surfacedirectory="$newmodeldir$surface"
    mkdir -p "$surfacedirectory"

}
