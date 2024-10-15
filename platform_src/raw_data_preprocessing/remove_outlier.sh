remove_outlier() {

    # spacer
    echo -e "\n"
    # outlier removal
    read -p "Enter the exact well to remove (e.g. B2) or 'r' to return: " outlier

    if [ "$outlier" == "r" ]; then
        echo "Returning.."

    else

                ## check if well designated is found in the csv
                chosen_dir=$chosen_dir
                outlier=$outlier
                column_name="Well"
                tidydata="Datasets/tidy_dataset.csv"
                tidydatapath=$chosen_dir$tidydata

                # Finding the column number based on column name
                column_number=$(head -1 "$tidydatapath" | awk -F, -v colname="$column_name" '
                    {
                        for (i = 1; i <= NF; i++) {
                            if ($i == colname) {
                                print i;
                                exit;
                            }
                        }
                    }'
                )

                if [ -z "$column_number" ]; then
                    echo "Column not found"
                    exit 1
                fi

                # Searching for the value in the identified column
                if awk -F, -v col="$column_number" -v value="$outlier" 'NR>1 && $col == value' "$tidydatapath" | grep -q .; then
                    
                    # spacer
                    echo -e "\n"
                    echo "$outlier found in column $column_name"
                    echo -e "\n"

                    if python3 "/app/analysis_scripts/2_outlier_removal.py" "$chosen_dir" "$outlier"; then
                        echo -e "Tidy dataset revised to remove outlier.."

                    else
                        echo -e "Removing outlier encountered an error. Aborting further execution."
                    fi
                    
                else
                    # spacer
                    echo -e "\n"
                    echo "Outlier not found in column $column_name"
                fi


    fi

    echo -e "\n"

}
