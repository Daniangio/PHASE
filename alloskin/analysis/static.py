"""
Goal 1 (Corrected): Intrinsic Dimension + State Sensitivity (MI/JSD/MMD/AUC/KL)

This replaces Information Imbalance, which is mathematically invalid when
Y is a binary Active/Inactive label.

Available state metrics:
    "mi"   – Mutual Information
    "jsd"  – Jensen–Shannon Divergence
    "mmd"  – Maximum Mean Discrepancy (Gaussian kernel)
    "kl"   – Symmetrized KL Divergence
    "auc"  – Logistic Regression AUC
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, Callable
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

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
# --- Helper functions for the different state-sensitivity metrics
# -------------------------------------------------------------------------

def estimate_jsd(pA, pI, bins=40):
    """Jensen-Shannon divergence between two multidimensional distributions."""
    histA, _ = np.histogramdd(pA, bins=bins, density=True)
    histI, _ = np.histogramdd(pI, bins=bins, density=True)
    histA += 1e-12
    histI += 1e-12
    m = 0.5 * (histA + histI)
    return 0.5 * entropy(histA, m) + 0.5 * entropy(histI, m)


def estimate_mmd(pA, pI, sigma=0.5):
    """Gaussian-kernel MMD estimate."""
    def kernel(x, y):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))
    m = len(pA)
    n = len(pI)
    xx = np.mean([kernel(pA[i], pA[j]) for i in range(m) for j in range(m)])
    yy = np.mean([kernel(pI[i], pI[j]) for i in range(n) for j in range(n)])
    xy = np.mean([kernel(pA[i], pI[j]) for i in range(m) for j in range(n)])
    return xx + yy - 2 * xy


def estimate_symmetric_kl(pA, pI, bandwidth=0.1):
    """Symmetrized KL divergence via KDE."""
    kdeA = KernelDensity(bandwidth=bandwidth).fit(pA)
    kdeI = KernelDensity(bandwidth=bandwidth).fit(pI)

    logA_A = kdeA.score_samples(pA)
    logI_A = kdeI.score_samples(pA)
    logA_I = kdeA.score_samples(pI)
    logI_I = kdeI.score_samples(pI)

    KL_A_I = np.mean(logA_A - logI_A)
    KL_I_A = np.mean(logI_I - logA_I)
    return KL_A_I + KL_I_A


def estimate_auc(X, Y):
    """Logistic regression AUC using 5-fold stratified cross-validation."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    aucs = []
    for train, test in skf.split(X, Y):
        model = LogisticRegression(max_iter=200, solver="lbfgs").fit(X[train], Y[train])
        preds = model.predict_proba(X[test])[:, 1]
        aucs.append(roc_auc_score(Y[test], preds))
    return float(np.mean(aucs))


def compute_state_sensitivity(X, Y, method="mi"):
    """
    Computes a scalar sensitivity of residue coordinates X to binary state Y.

    Inputs:
        X: (N, F) features of residue over trajectory
        Y: (N,) binary labels (0/1)
    """

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Split into active/inactive
    XA = X[Y == 1]
    XI = X[Y == 0]

    if XA.shape[0] < 10 or XI.shape[0] < 10:
        return np.nan

    method = method.lower()

    if method == "mi":
        # Discretize using quantiles
        bins = 30
        X_disc = np.zeros(X.shape[0], dtype=int)
        for k in range(X.shape[1]):
            X_disc += np.digitize(X[:, k], np.quantile(X[:, k], np.linspace(0,1,bins))) * (k+1)
        return mutual_info_score(X_disc, Y)

    elif method == "jsd":
        return estimate_jsd(XA, XI)

    elif method == "mmd":
        return estimate_mmd(XA, XI)

    elif method == "kl":
        return estimate_symmetric_kl(XA, XI)

    elif method == "auc":
        return estimate_auc(X, Y)

    else:
        raise ValueError(f"Unknown state metric: {method}")


# -------------------------------------------------------------------------
# --- Worker: Intrinsic Dimension + State sensitivity
# -------------------------------------------------------------------------

def _static_worker_state(
    item: Tuple[str, np.ndarray],
    labels_Y: np.ndarray,
    n_samples: int,
    maxk: int,
    state_metric: str
) -> Tuple[str, Dict[str, float]]:

    res_key, features_3d = item
    results = {"id": np.nan, "id_error": np.nan, "state_score": np.nan}

    try:
        X = features_3d.reshape(n_samples, -1)

        # 1. Intrinsic Dimension (same as your original implementation)
        if DADAPY_AVAILABLE:
            data_obj = Data(coordinates=X, maxk=maxk, n_jobs=1)
            id_val, id_err, _ = data_obj.compute_id_2NN()
            results["id"] = id_val
            results["id_error"] = id_err

        # 2. State sensitivity (new, correct)
        score = compute_state_sensitivity(X, labels_Y, method=state_metric)
        results["state_score"] = score

    except Exception:
        pass

    return (res_key, results)


# -------------------------------------------------------------------------
# --- Main class
# -------------------------------------------------------------------------

class StaticStateSensitivity(BaseStaticReporter):
    """
    Goal 1 corrected:
        - Intrinsic Dimension (Dadapy)
        - State Sensitivity (MI/JSD/MMD/KL/AUC)
    
        Mutual Information (MI)
        Jensen–Shannon Divergence (JSD)
        KL Divergence (symmetrized)
        Maximum Mean Discrepancy (MMD)
        Logistic Classifier AUC
    """

    def _get_worker_function(self) -> Callable:
        return _static_worker_state

    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        maxk = kwargs.get("maxk", min(100, n_samples - 1))
        state_metric = kwargs.get("state_metric", "mi") # "mi" "jsd", "mmd", "kl", "auc"
        return dict(maxk=maxk, state_metric=state_metric)
