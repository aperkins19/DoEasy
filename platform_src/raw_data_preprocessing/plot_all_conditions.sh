
plot_all_conditions() {

    local plot_file_name="timeseries_of_raw_data_all_condtions.png"

    if python3 "/app/platform_src/raw_data_preprocessing/plot_all_conditions.py" "$chosen_dir" "$plot_file_name"; then
        :
    else
        echo -e "Replotting all wells encountered an error. Aborting further execution."
    fi
}

