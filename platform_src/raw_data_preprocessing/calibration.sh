calibration() {

    if python3 "/app/analysis_scripts/4_calibration.py" "$chosen_dir"; then
        echo -e "Calibration Complete."
    else
        echo -e "Calibration encountered an error. Aborting further execution."
    fi
}