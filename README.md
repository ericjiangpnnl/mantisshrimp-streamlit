# MantisShrimp Streamlit

A Streamlit application for photometric redshift estimation using the MantisShrimp computer vision model.

## Project Structure

The project follows Streamlit best practices with a modular structure:

```
MantisShrimp-Streamlit/
├── app/                      # Main application directory
│   ├── __init__.py           # Make it a proper package
│   ├── main.py               # Main Streamlit entry point
│   ├── config.py             # App configuration
│   ├── pages/                # Streamlit pages directory
│   │   ├── __init__.py
│   │   └── about.py          # About page
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── download.py       # Download utilities
│       └── visualization.py  # Visualization utilities
├── data/                     # Data directory
│   └── dustmaps/             # Dust maps data
├── images/                   # Images directory
├── mantis_shrimp/            # Core package
├── run.sh                    # Run script
└── requirements.txt          # Dependencies
```

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

```bash
# Make the script executable (if not already)
chmod +x run.sh

# Run the script
./run.sh
```

Or manually:

```bash
# Activate the conda environment
conda activate mantisshrimp

# Run the Streamlit app
streamlit run app/main.py
```

The application will be available at http://localhost:8501 by default.

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

## License

This application is subject to the same license as the original MantisShrimp project. See the LICENSE.txt file for details.
