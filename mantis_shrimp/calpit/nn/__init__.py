from .umnn import MonotonicNN  # noqa

# IsplineNN import is optional - only import if splinebasis is available
try:
    from .ispline_nn import IsplineNN  # noqa
except ImportError:
    # splinebasis dependency not available, IsplineNN functionality disabled
    pass
