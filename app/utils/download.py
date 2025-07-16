"""
Utility functions for downloading data from the MantisShrimp application.
"""
import os
import json
import base64
import tarfile

def get_download_link_for_json(data, filename):
    """
    Generate a download link for JSON data.
    
    Parameters:
    data (dict): The data to be saved as JSON.
    filename (str): The name of the file to be downloaded.
    
    Returns:
    str: HTML link for downloading the JSON file.
    """
    json_str = json.dumps(data)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="{filename}">Download JSON Results</a>'
    return href

def get_download_link_for_fits(tarfile_name, savepath):
    """
    Create a tarball of FITS files and generate a download link.
    
    Parameters:
    tarfile_name (str): The name of the tar file to be created.
    savepath (str): The directory where FITS files are stored.
    
    Returns:
    str: HTML link for downloading the tarball, or None if no files found.
    """
    filters = ['Galex', 'g', 'r', 'i', 'z', 'y', 'Unwise']
    filepaths = []
    
    # Collect all file paths for the specified filters
    for filter_name in filters:
        filename = os.path.join(savepath, f'0.{filter_name}.fits')
        if os.path.exists(filename):
            filepaths.append(filename)
    
    if not filepaths:
        return None
    
    # Create the tarfile path in the /tmp directory
    tarfile_path = os.path.join('/tmp', tarfile_name)
    with tarfile.open(tarfile_path, "w:gz") as tar:
        for filepath in filepaths:
            try:
                # Add the file to the tar file
                tar.add(filepath, arcname=os.path.basename(filepath))
            except Exception as e:
                import streamlit as st
                st.error(f"An error occurred while adding {filepath}: {e}")
    
    # Read the tar file and create a download link
    with open(tarfile_path, "rb") as file:
        bytes_data = file.read()
    
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/x-tar;base64,{b64}" download="{tarfile_name}">Download FITS Files</a>'
    return href
