
create_tidydataset() {
    
    # get variables
    local chosen_dir=$chosen_dir

    ###
    # generate tidy dataset

    if python3 "/app/platform_src/raw_data_preprocessing/clean_tidy_rawdata.py" "$chosen_dir"; then
        echo -e "Tidy dataset generated."
    else
        echo -e "Tidy data generation encountered an error. Aborting further execution."
        exit 0
    fi
}