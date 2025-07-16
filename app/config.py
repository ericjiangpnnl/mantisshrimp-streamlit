"""
Configuration settings for the MantisShrimp Streamlit application.
"""
import os

# Set up paths
SAVEPATH = '/tmp/mantis_shrimp_fits/'
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH, exist_ok=True)

# Device configuration
DEVICE = 'cpu'

# Dustmaps configuration
DUSTMAPS_DIR = './data/dustmaps/'
