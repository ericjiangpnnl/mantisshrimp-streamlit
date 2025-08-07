"""
Main entry point for the MantisShrimp Streamlit application.
"""
import os
import sys
import io
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uuid
import numpy as np
import streamlit as st
import torch
from astropy.coordinates import SkyCoord
import astropy.units as u
from werkzeug.utils import secure_filename

# Import from app modules
from app.config import SAVEPATH, DEVICE, DUSTMAPS_DIR
from app.utils.download import get_download_link_for_json, get_download_link_for_fits
from app.utils.visualization import is_number_with_decimal
from app.utils.downloads import ensure_all_required_files

# Ensure all required files are downloaded before proceeding
ensure_all_required_files()

# Configure dustmaps
from dustmaps.config import config
config.reset()
config['data_dir'] = DUSTMAPS_DIR

# Import mantis_shrimp modules
from mantis_shrimp import preprocess
from mantis_shrimp import models
from mantis_shrimp import augmentation
from mantis_shrimp import utils
from mantis_shrimp import pipeline

# Import CalPit from local package with detailed diagnostics
import_status = {}
CALPIT_AVAILABLE = True

# Test each import individually for debugging
try:
    from mantis_shrimp.calpit import CalPit
    import_status['CalPit'] = "✅ Success"
except ImportError as e:
    import_status['CalPit'] = f"❌ Failed: {str(e)}"
    CALPIT_AVAILABLE = False

try:
    from mantis_shrimp.calpit.nn.umnn import MonotonicNN
    import_status['MonotonicNN'] = "✅ Success"
except ImportError as e:
    import_status['MonotonicNN'] = f"❌ Failed: {str(e)}"
    CALPIT_AVAILABLE = False

try:
    from mantis_shrimp.calpit.utils import normalize
    import_status['normalize'] = "✅ Success"
except ImportError as e:
    import_status['normalize'] = f"❌ Failed: {str(e)}"
    CALPIT_AVAILABLE = False

# If CalPit is not available, define a simple normalize function as a fallback
if not CALPIT_AVAILABLE:
    def normalize(data, grid):
        """Simple normalization fallback when CalPit is not available"""
        normalized_data = data / np.sum(data, axis=-1, keepdims=True)
        return normalized_data

# Load model
@st.cache_resource
def load_model():
    """Load the model and class bins with caching for better performance"""
    return models.trained_early(DEVICE)

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
    background-color: transparent;
    padding: 10px;
    text-align: center;
    color: #333;
}
.footer .footer-inner {
    max-width: 960px;  
    margin: 0 auto;
    text-align: center;
    color: white;
    font-size: 0.9rem;
}
.accordion-header {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
model, CLASS_BINS_npy = load_model()

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

# System Diagnostics accordion
with st.expander("🔧 System Diagnostics (Debug Info)"):
    st.markdown("**CalPit Import Status:**")
    for component, status in import_status.items():
        st.write(f"• **{component}**: {status}")
    
    st.markdown("---")
    st.markdown("**System Information:**")
    st.write(f"• **Python Version**: {sys.version}")
    st.write(f"• **Current Working Directory**: {os.getcwd()}")
    st.write(f"• **Python Path**: {sys.path[:3]}...")  # Show first 3 paths
    
    # Check if CalPit directory exists
    calpit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mantis_shrimp', 'calpit')
    calpit_exists = os.path.exists(calpit_dir)
    st.write(f"• **CalPit Directory Exists**: {'✅ Yes' if calpit_exists else '❌ No'} ({calpit_dir})")
    
    if calpit_exists:
        # List contents of calpit directory
        try:
            calpit_contents = os.listdir(calpit_dir)
            st.write(f"• **CalPit Directory Contents**: {calpit_contents}")
        except Exception as e:
            st.write(f"• **CalPit Directory Contents**: ❌ Error reading: {e}")
    
    # Check for __init__.py files
    init_files = []
    for subdir in ['', 'nn', 'nn/umnn']:
        init_path = os.path.join(calpit_dir, subdir, '__init__.py')
        init_exists = os.path.exists(init_path)
        init_files.append(f"{subdir or 'root'}/__init__.py: {'✅' if init_exists else '❌'}")
    st.write(f"• **__init__.py Files**: {', '.join(init_files)}")
    
    st.markdown("---")
    st.write(f"**Overall CalPit Status**: {'✅ Available' if CALPIT_AVAILABLE else '❌ Not Available'}")

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
                st.stop()
        
        if user_dec < -30:
            st.error("DECLINATION TOO LOW, < -30 NOT ALLOWED")
            st.stop()
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get data
        status_text.text("Fetching data...")
        progress_bar.progress(10)
        
        data = pipeline.get_data(0, user_ra, user_dec, SAVEPATH)
        
        status_text.text("Processing data...")
        progress_bar.progress(40)
        
        photo_z, PDF, x = pipeline.evaluate(0, user_ra, user_dec, model=model, data=data, SAVEPATH=SAVEPATH, device=DEVICE)
        
        progress_bar.progress(70)
        
        # Apply CalPit if requested
        if use_calpit:
            status_text.text("Applying CalPit calibration...")
            
            monotonicnn_model = MonotonicNN(84, [1024, 1024, 1024, 1024], sigmoid=True)
            monotonicnn_model.to(DEVICE)
            calpit_model = CalPit(model=monotonicnn_model)
            
            normalized_feature_vector = pipeline.get_feature_vector(data)
            
            # Use relative path approach consistent with other model loading
            import os
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from app/
            calpit_path = os.path.join(current_dir, 'mantis_shrimp', 'MODELS_final', 'calpit_checkpoint.pt')
            ckpt = torch.load(calpit_path, map_location=torch.device(DEVICE), weights_only=True)
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
        fits_download_link = get_download_link_for_fits(tarfile_name, SAVEPATH)
        
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

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-inner">
        <p>&copy; PNNL. All rights reserved.</p>
    </div>
</div>
""", unsafe_allow_html=True)
