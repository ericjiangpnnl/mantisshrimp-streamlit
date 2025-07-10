# MantisShrimp Streamlit Application

This is a Streamlit version of the MantisShrimp application, which provides a web interface for photometric redshift estimation using a computer vision model.

## Overview

The MantisShrimp Streamlit application allows users to:

- Enter coordinates (RA and DEC) to analyze astronomical objects
- View visualizations of photometric redshift predictions
- Download results as JSON files and FITS files
- Access information about the project and its limitations

## Requirements

- Python 3.7+
- Streamlit 1.0+
- All dependencies from the original MantisShrimp project

## Installation

The application uses the same environment as the original MantisShrimp project. Make sure you have set up the environment according to the main README.md file.

```bash
# If you haven't already, install the MantisShrimp package
git clone https://github.com/pnnl/MantisShrimp.git
cd MantisShrimp
git lfs pull
conda install -n base -c conda-forge mamba
mamba env create --name mantisshrimp --file production.yml
conda activate mantisshrimp
pip install .
```

## Running the Application

There are two ways to run the Streamlit application:

### Option 1: Using the run script

```bash
# Make the script executable (if not already)
chmod +x run_streamlit_app.sh

# Run the script
./run_streamlit_app.sh
```

### Option 2: Manual execution

```bash
# Activate the conda environment
conda activate mantisshrimp

# Run the Streamlit app
streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501 by default.

## Dependencies

The application requires several Python packages to run. The core dependencies are listed in the `requirements-streamlit.txt` file. You can install them with:

```bash
pip install -r requirements-streamlit.txt
```

Note that some optional dependencies (sep_pjw, calpit, splinebasis) are not required for the application to run, but may enhance certain features if available. The application includes fallback implementations for these dependencies.

## Features

### Home Page

- Input form for entering coordinates
- Advanced settings for customizing the analysis
- Visualization of results
- Download options for results

### About Page

- Information about the project
- Usage instructions
- Technical details
- Limitations and considerations

## Differences from Flask Version

The Streamlit version offers several advantages over the original Flask application:

1. **Interactive UI**: More responsive and interactive user interface
2. **Progress Indicators**: Visual feedback during data processing
3. **Simplified Deployment**: Easier to run and deploy
4. **Enhanced Navigation**: Sidebar navigation for better user experience
5. **Responsive Layout**: Better support for different screen sizes

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed correctly
2. Check that the dustmaps directory is properly configured
3. Verify that the model files are available in the expected locations
4. Check the console for any error messages

## License

This application is subject to the same license as the original MantisShrimp project. See the main LICENSE.txt file for details.
