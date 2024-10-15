feature_extraction() {

    if python3 "/app/platform_src/raw_data_preprocessing/feature_extraction.py" "$chosen_dir"; then
        echo -e "Features Extracted."
    else
        echo -e "Feature extraction encountered an error. Aborting further execution."
    fi
}
