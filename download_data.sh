#!/bin/bash

# Download the data to a temp directory
mkdir -p ./tmp

# wget -O ./tmp/data.tar https://fileserver.wanglab.science/p_ml_cleanup/to_upload/ml_pipeline/ml_pipeline/input_data.tar    # TODO: Update the URL

# Extract the data
tar -xf ./tmp/data.tar -C ./data/

# Remove the temp dir
rm -rf ./tmp