"""
Download utilities for MantisShrimp Streamlit application.
Handles downloading of large files needed by the application.
"""
import os
import gdown
import streamlit as st


def ensure_planck_dustmap():
    """
    Download Planck dustmap file if not present locally.
    
    This function checks if the Planck dustmap FITS file exists locally,
    and if not, downloads it from Google Drive using gdown.
    
    Returns:
        str: Path to the Planck dustmap file
    """
    # Define the path where the dustmap should be stored
    dustmap_path = "mantis_shrimp/dustmaps/planck/HFI_CompMap_ThermalDustModel_2048_R1.20.fits"
    
    # Check if file already exists
    if os.path.isfile(dustmap_path):
        return dustmap_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dustmap_path), exist_ok=True)
    
    # Google Drive download URL
    url = "https://drive.google.com/uc?id=1wzLWeP1cFI1OUXazligXdKRuQ5WCUQzS"
    
    # Download with user feedback
    with st.spinner("Downloading Planck dustmap file (this may take a few minutes)..."):
        try:
            gdown.download(url, dustmap_path, quiet=False)
            
            # Verify the download was successful
            if os.path.isfile(dustmap_path):
                file_size = os.path.getsize(dustmap_path)
                size_mb = file_size / (1024 * 1024)
                
                if file_size > 1024:  # More than 1KB suggests successful download
                    st.success(f"✅ Successfully downloaded Planck dustmap ({size_mb:.1f} MB)")
                    return dustmap_path
                else:
                    raise Exception(f"Downloaded file is too small ({file_size} bytes)")
            else:
                raise Exception("File was not created after download")
                
        except Exception as e:
            # Clean up partial download
            if os.path.exists(dustmap_path):
                os.remove(dustmap_path)
            
            st.error(f"❌ Failed to download Planck dustmap: {str(e)}")
            st.error("Please check your internet connection and try again.")
            st.stop()
    
    return dustmap_path


def ensure_all_required_files():
    """
    Ensure all required large files are downloaded and available.
    
    This function should be called early in the app initialization
    to download any missing required files.
    """
    # Download Planck dustmap
    ensure_planck_dustmap()
    
    # Add other file downloads here as needed
    # e.g., model files, other dustmaps, etc.
