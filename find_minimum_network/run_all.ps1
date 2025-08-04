Write-Host "Running all minimum search jobs..."

python find_minimum_cli.py -n 2 -i 2_binarized.csv -o 2_binarized_output.csv -m binarized
python find_minimum_cli.py -n 2 -i 2_continuous.csv -o 2_continious_output.csv -m continuous
python find_minimum_cli.py -n 3 -i 3_continuous.csv -o 3_continious_output.csv -m continuous
python find_minimum_cli.py -n 3 -i 3_binarized.csv -o 3_binarized_output.csv -m binarized


Write-Host "All done."