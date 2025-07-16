"""
Visualization utilities for the MantisShrimp application.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

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

def create_custom_style():
    """
    Create and return custom styling for matplotlib visualizations.
    
    Returns:
    dict: Dictionary of style parameters.
    """
    return {
        'figure.figsize': (12, 8),
        'axes.facecolor': '#f5f5f5',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 12
    }

def apply_custom_style(fig, ax):
    """
    Apply custom styling to a matplotlib figure and axes.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure to style.
    ax (matplotlib.axes.Axes): The axes to style.
    
    Returns:
    tuple: The styled figure and axes.
    """
    # Apply custom styling
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f5f5f5')
    
    # Improve readability
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    return fig, ax
