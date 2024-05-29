
plot_all_conditions() {

    local plot_file_name="timeseries_subplots_of_raw_data_all_condtions.png"

    if python3 "/app/analysis_scripts/plot_all_raw_data_subplots.py" "$chosen_dir" "$plot_file_name"; then
        :
    else
        echo -e "Replotting all wells encountered an error. Aborting further execution."
    fi
}

