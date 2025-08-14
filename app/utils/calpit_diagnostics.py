"""
CalPit diagnostics utility for debugging import issues.

This module provides diagnostic functions to help debug CalPit import problems
in the Streamlit deployment environment.
"""
import os
import sys
import streamlit as st


def show_calpit_diagnostics(import_status, CALPIT_AVAILABLE):
    """
    Display CalPit system diagnostics panel for debugging import issues.
    
    Args:
        import_status (dict): Dictionary containing import status for each CalPit component
        CALPIT_AVAILABLE (bool): Overall CalPit availability status
    """
    with st.expander("🔧 CalPit System Diagnostics (Debug Info)"):
        st.markdown("**CalPit Import Status:**")
        for component, status in import_status.items():
            st.write(f"• **{component}**: {status}")
        
        st.markdown("---")
        st.markdown("**System Information:**")
        st.write(f"• **Python Version**: {sys.version}")
        st.write(f"• **Current Working Directory**: {os.getcwd()}")
        st.write(f"• **Python Path**: {sys.path[:3]}...")  # Show first 3 paths
        
        # Check if CalPit directory exists
        calpit_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'mantis_shrimp', 'calpit')
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


def test_calpit_imports():
    """
    Test CalPit imports individually and return status information.
    
    Returns:
        tuple: (import_status dict, CALPIT_AVAILABLE bool)
    """
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

    return import_status, CALPIT_AVAILABLE
