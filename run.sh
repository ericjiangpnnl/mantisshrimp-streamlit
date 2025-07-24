#!/bin/bash

# Activate the conda environment
echo "Activating mantisshrimp conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mantisshrimp

# Run the Streamlit application
echo "Starting Streamlit application..."
streamlit run app/1_Home.py

# Note: If the application doesn't start, make sure you have installed all dependencies
# You can install them with: pip install -r requirements.txt
