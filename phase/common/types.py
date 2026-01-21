"""
Common type hints used across the phase library.
"""

from typing import Dict
import numpy as np
import MDAnalysis as mda

# A dictionary mapping a residue key (e.g., "res_50")
# to its feature array (N_frames, N_features)
FeatureDict = Dict[str, np.ndarray]

# The trajectory object type
TrajectoryObject = mda.Universe