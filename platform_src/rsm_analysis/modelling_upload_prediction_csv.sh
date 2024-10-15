#! /bin/bash

upload_prediction_csv() {

    echo -e "\n"
    echo -e "To Upload a file, open the directory and place your .csv file into '/Place Input Files Here'"
    echo -e "Warning: make sure your file is saved somewhere else as it will be deleted from '/Place Input Files Here' after the operation."
    echo -e "Once you have done that, return and hit enter to continue."
    read  # wait user to hit enter
    echo -e "\n"

    # verify csv
    # Set the directory to check
    inputbox="/app/Place Input Files Here/"
    file_destination="${select_model_dir%/}/prediction/"

    # Find all .csv files in the directory and store them in an array
    while IFS= read -r -d $'\0' file; do
        csv_files+=("$file")
    done < <(find "$inputbox" -maxdepth 1 -name "*.csv" -print0)

    # Get the count of .csv files
    csv_count=${#csv_files[@]}

    # Check if the count is exactly 1
    if [ "$csv_count" -eq 1 ]; then

        # Extract filename from the path
        filename=$(basename "${csv_files[0]}")

        # Move the file
        echo -e "One .csv file found: ${filename}, saving to project.."
        # mkdir if doesn't exist
        if [ -d $file_destination ]; then
            echo "/prediction Directory already exists."
        else
            echo "Creating directory: $file_destination"
        fi

        mv "${csv_files[0]}" "${file_destination}input_prediction_dataset.csv"

        # Check if the move was successful
        if [ -f "$file_destination/input_prediction_dataset.csv" ]; then
            echo -e "File successfully moved."
        else
            echo -e "Failed to move the file."
        fi


    else
        if [ "$csv_count" -eq 0 ]; then
            echo "No .csv files found in the directory, please place one in '/Place Input Files Here' and try again."
        else
            echo -e  "There are multiple (.csv) files in the directory. Number: ${csv_count}"
            # Debugging: List all files in the directory
            echo "Listing all files in the directory:"
            ls "$inputbox"
            echo -e "Please ensure only the file to be uploaded is in '/Place Input Files Here' and try again."
        fi
    fi
}