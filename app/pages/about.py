"""
About page for the MantisShrimp Streamlit application.
"""
import streamlit as st

def show():
    """Display the About page content."""
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
