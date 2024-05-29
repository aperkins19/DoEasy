#! /bin/bash

upload_raw_data_csv() {

    echo -e $chosen_dir

    echo -e "\n"
    echo -e "To Upload a file, open the directory and place your .csv file into '/Place Input Files Here'"
    echo -e "Warning: make sure your file is saved somewhere else as it will be deleted from '/Place Input Files Here' after the operation."
    echo -e "Once you have done that, return and hit enter to continue."
    read  # wait user to hit enter
    echo -e "\n"


    # verify csv
    # Set the directory to check
    inputbox="/app/Place Input Files Here/"
    project_destination="${chosen_dir%/}/Datasets/RawData/"

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
        mv "${csv_files[0]}" "${project_destination}/input_raw_data_dataset.csv"

        tidydata_answer=""

        # Loop until the user provides a valid response (y or n)
        while [[ "$tidydata_answer" != "y" && "$tidydata_answer" != "n" ]]; do
            # ask if the dataset is already a tidy dataset
            read -p "Is the raw data already in the tidydata format? (y/n)" tidydata_answer
            echo -e "\n"

            if [ "$tidydata_answer" = "y" ]; then
                echo -e "The dataset is already in the tidy format."
                cp "${project_destination}/input_raw_data_dataset.csv" $chosen_dir/Datasets/tidy_dataset.csv
                echo -e "There is no need to conduct data preprocessing, continue straight to response variable analysis."


            elif [ "$tidydata_answer" = "n" ]; then
                echo "The dataset is not in the tidy format."
                echo "Conduct preprocessing to generate tidy data before progressing."
            else
                echo "Invalid input. Please enter 'y' for yes or 'n' for no."
            fi
        done
        

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