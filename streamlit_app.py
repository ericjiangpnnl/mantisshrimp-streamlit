import os
import io
import base64
import json
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib import gridspec
import time
import uuid
from werkzeug.utils import secure_filename

# Configure dustmaps
from dustmaps.config import config
config.reset()
config['data_dir'] = './data/dustmaps/'

# Import mantis_shrimp modules
from mantis_shrimp import preprocess
from mantis_shrimp import models
from mantis_shrimp import augmentation
from mantis_shrimp import utils
from mantis_shrimp import pipeline

# Import CalPit if available
try:
    from calpit import CalPit
    from calpit.nn.umnn import MonotonicNN
    from calpit.utils import normalize
    # Try to import a module that might be missing
    import splinebasis
    CALPIT_AVAILABLE = True
except ImportError:
    # If any import fails, mark CalPit as unavailable
    CALPIT_AVAILABLE = False
    # Define a simple normalize function as a fallback
    def normalize(data, grid):
        """Simple normalization fallback when CalPit is not available"""
        normalized_data = data / np.sum(data, axis=-1, keepdims=True)
        return normalized_data

# Set up paths and device
SAVEPATH = '/tmp/mantis_shrimp_fits/'
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH, exist_ok=True)
device = 'cpu'

# Load model
@st.cache_resource
def load_model():
    """Load the model and class bins with caching for better performance"""
    return models.trained_early(device)

# Helper function to check if input is a number with decimal
def is_number_with_decimal(s):
    """
    Checks if a given string can be converted to a floating-point number.
    
    Parameters:
    s (str): The string to check.
    
    Returns:
    bool: True if the string can be converted to a floating-point number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

# Function to create download link for JSON data
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

# Function to create download link for FITS files
def get_download_link_for_fits(tarfile_name):
    """
    Create a tarball of FITS files and generate a download link.
    
    Parameters:
    tarfile_name (str): The name of the tar file to be created.
    
    Returns:
    str: HTML link for downloading the tarball.
    """
    filters = ['Galex', 'g', 'r', 'i', 'z', 'y', 'Unwise']
    filepaths = []
    
    # Collect all file paths for the specified filters
    for filter_name in filters:
        filename = os.path.join(SAVEPATH, f'0.{filter_name}.fits')
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
                st.error(f"An error occurred while adding {filepath}: {e}")
    
    # Read the tar file and create a download link
    with open(tarfile_path, "rb") as file:
        bytes_data = file.read()
    
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/x-tar;base64,{b64}" download="{tarfile_name}">Download FITS Files</a>'
    return href

# Main function
def main():
    # Set page config
    st.set_page_config(
        page_title="Mantis Shrimp Redshift Webapp",
        page_icon="🦐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f5f5f5;
        padding: 10px;
        text-align: center;
    }
    .accordion-header {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About"])
    
    # Load model
    model, CLASS_BINS_npy = load_model()
    
    if page == "Home":
        # Header
        st.title("Mantis Shrimp Redshift Webapp")
        st.header("Neural Network Prediction")
        
        # Advanced settings
        show_advanced = st.checkbox("Show Advanced Settings")

        # Create form
        with st.form(key="input_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                name = st.text_input("Name (optional)", "")
            
            with col2:
                ra = st.text_input("RA (decimal degrees)", "")
            
            with col3:
                dec = st.text_input("DEC (decimal degrees)", "")
            
            if show_advanced:
                st.subheader("Advanced Settings")
                
                # Use a container to ensure the advanced settings are visible
                with st.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        use_calpit = st.checkbox("Calibrate Outputs With CalPit", value=False)
                        if use_calpit and not CALPIT_AVAILABLE:
                            st.warning("CalPit is not available. Please install it to use this feature.")
                            use_calpit = False
                    
                    with col2:
                        normalize_output = st.checkbox("Normalize Outputs (Switching This Off Produces PMFs)", value=False)
            else:
                use_calpit = False
                normalize_output = False
            
            submit_button = st.form_submit_button(label="Predict")
        
        # Limitations accordion
        with st.expander("Limitations"):
            st.markdown("""
            This computer vision model was trained on available spectroscopic data prior to major DESI releases (includes DESI early data release) -- that is to say, it was trained on (and evaluated with) a biased sample of galaxies. We are combining data from most major spectroscopic campaigns, including auxillary programs in SDSS E/BOSS that do not follow the standard targeting of giant red elliptical galaxies. This has some unknown effects: while it is well known that neural networks do not extrapolate well, it is unknown exactly how well they interpolate in low-density areas of feature space.
            
            Rather than limit what the user can request, we have opted to allow our network to perform inference at any location within the footprint of our telescopes-- even if the pointing does not have a discernable galaxy, if the galaxy would be considered outside our training sample, if the galaxy is blended with another object, or if the galaxy is obscured heavily by galactic dust from the milky-way. WE HAVE PLACED THE RESPONSIBILITY TO YOU, the user, to consider the context of the cutouts when evaluating whether our predictions should be trusted. If you can not discern your target galaxy in the center of the image, then our neural network will not provide an accurate photometric redshift.
            
            That being said, many pointing do not have any discernable flux in the Galex:UV bands. That is normal and reflective of our training set. Its a good rule of thumb to make sure that the galaxy is visible in the optical bands atleast.
            """)
        
        # Process form submission
        if submit_button and ra and dec:
            try:
                # Validate RA and DEC coordinates
                if is_number_with_decimal(ra) and is_number_with_decimal(dec):
                    user_ra = float(ra)
                    user_dec = float(dec)
                else:
                    try:
                        user_ra = float(SkyCoord(ra, dec).ra.degree)
                        user_dec = float(SkyCoord(ra, dec).dec.degree)
                    except ValueError as e:
                        st.error("INVALID COORDINATES PROVIDED")
                        return
                
                if user_dec < -30:
                    st.error("DECLINATION TOO LOW, < -30 NOT ALLOWED")
                    return
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get data
                status_text.text("Fetching data...")
                progress_bar.progress(10)
                
                data = pipeline.get_data(0, user_ra, user_dec, SAVEPATH)
                
                status_text.text("Processing data...")
                progress_bar.progress(40)
                
                photo_z, PDF, x = pipeline.evaluate(0, user_ra, user_dec, model=model, data=data, SAVEPATH=SAVEPATH, device=device)
                
                progress_bar.progress(70)
                
                # Apply CalPit if requested
                if use_calpit:
                    status_text.text("Applying CalPit calibration...")
                    
                    monotonicnn_model = MonotonicNN(84, [1024, 1024, 1024, 1024], sigmoid=True)
                    monotonicnn_model.to(device)
                    calpit_model = CalPit(model=monotonicnn_model)
                    
                    normalized_feature_vector = pipeline.get_feature_vector(data)
                    
                    ckpt = torch.load('./mantis_shrimp/MODELS_final/calpit_checkpoint.pt', map_location=torch.device(device), weights_only=True)
                    calpit_model.model.load_state_dict(ckpt)
                    calpit_model.model.train(False)
                    
                    y_grid = np.linspace(0, 1.6, 400)
                    PDF = PDF[None, :]
                    
                    cde_test = normalize((PDF.copy() + 1e-2) / np.sum(PDF.copy() + 1e-2, axis=1)[:, None], np.linspace(0, 1.6, 400))
                    new_cde = calpit_model.transform(normalized_feature_vector, cde_test, y_grid, batch_size=16)
                    new_cde = (new_cde.copy()) / np.sum(new_cde.copy(), axis=1)[:, None]
                    
                    # Update photoz based off the new cde
                    photo_z = np.sum(new_cde * CLASS_BINS_npy)
                    
                    # Normalize if requested
                    if normalize_output:
                        new_cde = normalize(new_cde, np.linspace(0, 1.6, 400))
                    
                    fig = pipeline.visualization(photo_z, new_cde.squeeze(), x, 0, user_ra, user_dec, CLASS_BINS_npy)
                else:
                    if normalize_output:
                        new_cde = normalize(PDF, np.linspace(0, 1.6, 400))
                    else:
                        new_cde = PDF
                    
                    fig = pipeline.visualization(photo_z, new_cde, x, 0, user_ra, user_dec, CLASS_BINS_npy)
                
                status_text.text("Generating visualization...")
                progress_bar.progress(90)
                
                # Display the figure
                st.pyplot(fig)
                
                # Prepare data for saving and download
                data_dict = {
                    'RA': user_ra,
                    'DEC': user_dec,
                    'photo_z': photo_z.tolist(),
                    'CDE': new_cde.tolist(),
                    'CalPit': use_calpit,
                    'PDF': normalize_output
                }
                
                if name:
                    unique_filename = secure_filename(name + '.json')
                else:
                    unique_filename = f'RA_{user_ra:.5f}__DEC_{user_dec:.5f}.json'
                
                tarfile_name = f'RA_{user_ra:.5f}__DEC_{user_dec:.5f}.tar'
                
                # Create download links
                json_download_link = get_download_link_for_json(data_dict, unique_filename)
                fits_download_link = get_download_link_for_fits(tarfile_name)
                
                # Display download links
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(json_download_link, unsafe_allow_html=True)
                with col2:
                    if fits_download_link:
                        st.markdown(fits_download_link, unsafe_allow_html=True)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif page == "About":
        st.title("About Mantis Shrimp")
        
        st.markdown("""
        ## Paper
        
        The Mantis Shrimp paper has been submitted for consideration as a journal article. You can find updated information about the paper on our [github repository](https://github.com/pnnl/MantisShrimp).
        
        ## Use
        
        The Webapp is a simple interface built using Streamlit; simply put in a name (used to tag the output files) a right ascension, and a declination. The webapp will query cutout services from external providers, prepare the data, then run those cutouts through a computer vision model. The output of that computer vision models is interpreted as a PDF (or PMF) allowing the extraction of point estimates plus errors, and could be used as a prior for downstream inference.
        
        Simply enter a RA and DEC in decimal degree form, and receive a photometric redshift!
        
        ## Advanced
        
        In our advanced settings, we allow a user to specify whether or not to apply CalPIT recalibration to the resulting PDFs. CalPIT is a python package that uses a neural network to calibrate outputs of other networks performing conditional density estimation. We observed that for large swaths of feature space, CalPIT was effective at decreasing the statistical bias of our model. It is our oppinion more work needs to go into this field of calibration for machine learning uncertainty quantification, but the technical debt in this specific task will also soon be alleviated by more complete spectroscopic surveys.  
        
        We also provide a toggle between probability density function (PDF) output and probability mass function (PMF) output. In PDF mode, the entire area underneath the curve represented by the function must integrate to a value of 1. This is achieved through normalization of an interpolation of the output of the model. More natively, our model outputs a PMF, where the model output represents the probability that the given galaxy has a redshift in a narrow bin of redshift. Both options will retain the same overall "Shape" of the PDF and do not affect point or uncertainty estimates.
        
        ## Limitations
        
        This computer vision model was trained on available spectroscopic data prior to major DESI releases (includes DESI early data release) -- that is to say, it was trained on (and evaluated with) a biased sample of galaxies. We are combining data from most major spectroscopic campaigns, including auxillary programs in SDSS E/BOSS that do not follow the standard targeting of giant red elliptical galaxies. This has some unknown effects: while it is well known that neural networks do not extrapolate well, it is unknown exactly how well they interpolate in low-density areas of feature space.
        
        Rather than limit what the user can request, we have opted to allow our network to perform inference at any location within the footprint of our telescopes-- even if the pointing does not have a discernable galaxy, if the galaxy would be considered outside our training sample, if the galaxy is blended with another object, or if the galaxy is obscured heavily by galactic dust from the milky-way. WE HAVE PLACED THE RESPONSIBILITY TO YOU, the user, to consider the context of the cutouts when evaluating whether our predictions should be trusted. If you can not discern your target galaxy in the center of the image, then our neural network will not provide an accurate photometric redshift.
        
        That being said, many pointing do not have any discernable flux in the Galex:UV bands. That is normal and reflective of our training set. Its a good rule of thumb to make sure that the galaxy is visible in the optical bands atleast.
        
        ## Authors
        
        * Andrew Engel, The Ohio State University and Pacfic Northwest National Laboratory
        * Nell Byler, Pacific Northwest National Laboratory
        * Adam Tsou, John Hopkins University
        * Gautham Narayan, University of Illinois Urbana-Champaign
        * Emmanuel Bonilla, Pacific Northwest National Laboratory
        * Ian Smith, Pacific Northwest National Laboratory
        
        ## Contact
        
        Please contact Andrew Engel for questions re. the project: andrew.william.engel@gmail.com
        
        ## Availability
        
        Servers have cost: we expect PNNL will be able to fund this webapp until September 2025 thanks to a grant from NASA. After that, the webapp will need to be transferred to an external host.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>&copy; PNNL. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
