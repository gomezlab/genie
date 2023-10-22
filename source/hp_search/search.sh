# Define the drugs, outcomes, and data types to search
drugs=('folfox' 'folfiri')
outcomes=('OS')
data_types=('comb' 'mut' 'cna' 'clin')

# Loop through the drugs, outcomes, and data types and run the scripts
for drug in "${drugs[@]}"; do
    for outcome in "${outcomes[@]}"; do
        for data_type in "${data_types[@]}"; do
            python3 search_rf_cv.py --drug "$drug" --outcome "$outcome" --data_type "$data_type" &
            python3 search_xgb_cv.py --drug "$drug" --outcome "$outcome" --data_type "$data_type" &
            wait
        done
    done
done