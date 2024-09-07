#!/bin/bash

# Define the output file
output_file="combined_metadata_V1.json"

# Initialize the output file
echo "[" > $output_file

# Counter to check if it's the first file
first=true

# Function to process a set of directories
process_dirs() {
  for r in "$@"; do
    facts=15
    increment=10
    while [[ $facts -le 1100 ]]; do
      dir="logs/pythia-1b-made-up-facts-r=$r-num-of-facts=$facts"
      # Check if the file exists
      if [[ -f "$dir/experiment_metadata.json" ]]; then
        # Add a comma before appending new JSON if it's not the first file
        if [[ $first == false ]]; then
          echo "," >> $output_file
        fi

        # Append the contents of the JSON file
        cat "$dir/experiment_metadata.json" >> $output_file
        first=false
      else
        echo "Warning: $dir/experiment_metadata.json not found."
      fi
      # Update facts value for next iteration
      if [[ $facts -eq 15 ]]; then
        facts=25
      else
        facts=$((facts + 25))
      fi
    done
  done
}

# Process pythia-1b files with different r values
process_dirs 1 2 4 8 16 32 64 128 256 512 1024 2028

# Close the JSON array
echo "]" >> $output_file

echo "All JSON files have been combined into $output_file"
