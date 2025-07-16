# Use the lightweight Micromamba base image
FROM mambaorg/micromamba:2.0-ubuntu22.04

# Set environment variables
ENV ENV_NAME=basic

# Update packages and Install packages
USER root

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y git python3 python3-pip wget git-lfs

COPY mantis_shrimp/dustmaps/csfd/ /env/mantis_shrimp/dustmaps/csfd/
COPY mantis_shrimp/dustmaps/planck/ /env/mantis_shrimp/dustmaps/planck/

# Set the working directory to the user's home (writable)
WORKDIR /home/mambauser

# Copy only the environment file first to optimize caching
COPY --chown=mambauser:mambauser production-streamlit.yml /home/mambauser/production-streamlit.yml

# Switch to the default non-root user provided by Micromamba
USER mambauser

# Create the environment in the writable user directory
# Create the environment and install packages in the writable user directory
RUN micromamba create -y -n $ENV_NAME && \
    micromamba install -y -n $ENV_NAME -c conda-forge --file /home/mambauser/production-streamlit.yml && \
    micromamba update && micromamba clean --all && micromamba clean --force-pkgs-dirs -y

# Copy the rest of the application (with correct permissions)
COPY --chown=mambauser:mambauser . /home/mambauser/env

# Install the application using the newly created environment
RUN micromamba run -n $ENV_NAME pip install -e /home/mambauser/env

# Set the working directory to the user's home (writable)
WORKDIR /home/mambauser/env

# Expose port 8501 for the Streamlit application
EXPOSE 8501

# Set the entrypoint to activate the environment and start the Streamlit server
ENTRYPOINT ["/bin/bash", "-c", "micromamba run -n basic streamlit run streamlit_app.py"]
