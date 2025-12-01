"""
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, Callable

# Limit threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    from dadapy.data import Data
    DADAPY_AVAILABLE = True
except ImportError:
    DADAPY_AVAILABLE = False

try:
    from .components import BaseStaticReporter
except ImportError:
    from alloskin.analysis.components import BaseStaticReporter


# -------------------------------------------------------------------------
# --- Main class
# -------------------------------------------------------------------------

class TICASlowModes(BaseStaticReporter):
    """
    """

    def _get_worker_function(self) -> Callable:
        return ...

    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        return dict()
