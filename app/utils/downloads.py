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


def ensure_model_files():
    """
    Download all required model files if not present locally.
    
    Downloads the neural network model weights and calibration files
    needed for redshift prediction.
    
    Returns:
        dict: Dictionary with paths to all model files
    """
    model_files = {
        'best_early': 'mantis_shrimp/MODELS_final/best_early.pt',
        'calpit_checkpoint': 'mantis_shrimp/MODELS_final/calpit_checkpoint.pt',
        'calpit_mean': 'mantis_shrimp/MODELS_final/calpit_stats/calpit_mean.npy',
        'calpit_std': 'mantis_shrimp/MODELS_final/calpit_stats/calpit_std.npy'
    }
    
    model_urls = {
        'best_early': 'https://drive.google.com/uc?id=1ZS3gxEVTKYuUP6RHJ6kLCn1rRYcs7EqK',
        'calpit_checkpoint': 'https://drive.google.com/uc?id=12Lnp7EUbwL6u72xr-ZLjxh3rx1sXlP0t',
        'calpit_mean': 'https://drive.google.com/uc?id=1XBfMS6VtEaXhku5mDjh7t3XznpRHFZ0W',
        'calpit_std': 'https://drive.google.com/uc?id=1MGDFnSEJD9TcI7QMxwUpqkW2ylI2XRiz'
    }
    
    downloaded_files = {}
    
    for file_key, file_path in model_files.items():
        # Check if file already exists
        if os.path.isfile(file_path):
            downloaded_files[file_key] = file_path
            continue
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download with user feedback
        with st.spinner(f"Downloading {file_key} model file..."):
            try:
                gdown.download(model_urls[file_key], file_path, quiet=False)
                
                # Verify the download was successful
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    size_mb = file_size / (1024 * 1024)
                    
                    if file_size > 100:  # More than 100 bytes suggests successful download
                        st.success(f"✅ Successfully downloaded {file_key} ({size_mb:.1f} MB)")
                        downloaded_files[file_key] = file_path
                    else:
                        raise Exception(f"Downloaded file is too small ({file_size} bytes)")
                else:
                    raise Exception("File was not created after download")
                    
            except Exception as e:
                # Clean up partial download
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                st.error(f"❌ Failed to download {file_key}: {str(e)}")
                st.error("Please check your internet connection and try again.")
                st.stop()
    
    return downloaded_files


def ensure_all_required_files():
    """
    Ensure all required large files are downloaded and available.
    
    This function should be called early in the app initialization
    to download any missing required files.
    """
    # Download Planck dustmap
    ensure_planck_dustmap()
    
    # Download model files
    ensure_model_files()
