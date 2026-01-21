"""Static state sensitivity analysis.

This module implements a *classifier-based* readout of how predictive each
residue is of the active / inactive state, together with a simple intrinsic
dimension estimate.

Compared to the previous MI-based implementation, this version:

    * Uses the full multi-dimensional descriptor for each residue
      (e.g. sin/cos of phi, psi, chi1) rather than averaging 1D MI values.
    * Reports a normalized state score in [0, 1] based on either
      cross-validated AUC (default) or, optionally, a cross-entropy based
      mutual-information surrogate.
    * Keeps an intrinsic-dimension (ID) estimate per residue using a
      PCA-variance criterion, which can be used as a downstream filter.

The public entry point is ``StaticStateSensitivity.run((features, labels_Y))``,
where:

    features : FeatureDict
        Mapping ``residue_id -> (n_frames, d)`` array.
    labels_Y : ndarray, shape (n_frames,)
        Binary labels, 1 = active, 0 = inactive.

The return value is a dict:

    { residue_id: { "id": float,
                    "state_score": float,
                    "score_type": "AUC" or "CE",
                    "auc": float,
                    "cross_entropy": float } }

This is plug-compatible with the existing runner.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Callable

# Optional: circular transform helper (not strictly needed here, but imported
# to keep backwards compatibility for users that relied on it externally).
try:  # pragma: no cover - optional dependency
    from phase.features.extraction import transform_to_circular  # noqa: F401
except Exception:  # pragma: no cover - defensive
    transform_to_circular = None  # type: ignore

from .components import BaseStaticReporter, FeatureDict  # :contentReference[oaicite:2]{index=2}

# --- Optional heavy dependencies (scikit-learn) ---
try:  # pragma: no cover - heavy dependency
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, log_loss
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    SKLEARN_AVAILABLE = False


def _estimate_intrinsic_dimension_pca(
    X: np.ndarray,
    variance_threshold: float = 0.9,
) -> float:
    """Estimate intrinsic dimensionality via PCA variance threshold.

    Parameters
    ----------
    X
        Array of shape (n_samples, d).
    variance_threshold
        Fraction of total variance to explain (e.g. 0.9). The ID is the
        smallest number of principal components whose cumulative explained
        variance exceeds this threshold.

    Returns
    -------
    float
        Estimated intrinsic dimension in [1, d]. Returns ``np.nan`` if the
        estimate is numerically unstable (e.g. no variance).
    """
    X = np.asarray(X)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)

    n_samples, d = X.shape
    if n_samples < 5 or d == 0:
        return float("nan")

    # Center the data
    Xc = X - X.mean(axis=0, keepdims=True)

    # Robust covariance (fall back to identity on failure)
    try:
        cov = np.cov(Xc, rowvar=False)
    except Exception:
        return float("nan")

    try:
        # Use symmetric eigensolver
        evals = np.linalg.eigvalsh(cov)
    except Exception:
        return float("nan")

    evals = np.asarray(evals, dtype=float)
    evals = np.clip(evals, 0.0, None)  # numerical noise

    total = float(evals.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")

    # Sort descending and compute cumulative explained variance
    evals_sorted = np.sort(evals)[::-1]
    cumsum = np.cumsum(evals_sorted) / total

    # Index of first component where cumulative variance >= threshold
    idx = int(np.searchsorted(cumsum, variance_threshold))
    # Convert 0-based index to 1-based dimension, but never exceed d
    dim = min(d, max(1, idx + 1))
    return float(dim)


def _static_worker_state(
    item,
    *,
    labels_Y: np.ndarray,
    n_samples: int,
    metric: str = "auc",
    n_splits: int = 5,
    random_state: int = 0,
    id_variance_threshold: float = 0.9,
) -> tuple[str, Dict[str, Any]]:
    """Worker that computes ID + state-sensitivity for a single residue.

    Parameters
    ----------
    item
        Tuple (residue_key, feature_array) from the FeatureDict.
    labels_Y
        Binary labels (1 = active, 0 = inactive), shape (n_samples,).
    n_samples
        Number of frames.
    metric
        Either "auc" (default) or "ce" (cross-entropy-based MI surrogate).
    n_splits
        Maximum number of CV folds for the classifier.
    random_state
        RNG seed for StratifiedKFold and the logistic regression.
    id_variance_threshold
        Variance threshold used in the PCA-based ID estimator.

    Returns
    -------
    (residue_key, metrics_dict)
    """
    res_key, X = item

    X = np.asarray(X)
    Y = np.asarray(labels_Y)

    # Coerce features to 2D: (n_samples, d)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if X.shape[0] != n_samples:
        raise ValueError(
            f"Residue {res_key!r}: feature length {X.shape[0]} does not match labels {n_samples}."
        )

    # Estimate intrinsic dimension first (independent of classifier)
    id_est = _estimate_intrinsic_dimension_pca(
        X,
        variance_threshold=id_variance_threshold,
    )

    # If sklearn is not available, we can only return ID.
    if not SKLEARN_AVAILABLE:  # pragma: no cover - defensive path
        return str(res_key), {
            "id": float(id_est),
            "state_score": float("nan"),
            "score_type": metric.upper(),
            "auc": float("nan"),
            "cross_entropy": float("nan"),
        }

    # Guard: must have at least 2 samples per class.
    XA = X[Y == 1]
    XI = X[Y == 0]
    if XA.shape[0] < 2 or XI.shape[0] < 2:
        return str(res_key), {
            "id": float(id_est),
            "state_score": float("nan"),
            "score_type": metric.upper(),
            "auc": float("nan"),
            "cross_entropy": float("nan"),
        }

    # ------------------------------------------------------------------
    # Cross-validated logistic classifier
    # ------------------------------------------------------------------
    # Determine effective number of folds based on smallest class.
    class_counts = np.bincount(Y.astype(int), minlength=2)
    min_class = int(class_counts.min())
    if min_class < 2:
        # Practically impossible after the above check, but be safe.
        return str(res_key), {
            "id": float(id_est),
            "state_score": float("nan"),
            "score_type": metric.upper(),
            "auc": float("nan"),
            "cross_entropy": float("nan"),
        }

    eff_splits = max(2, min(n_splits, min_class))

    cv = StratifiedKFold(
        n_splits=eff_splits,
        shuffle=True,
        random_state=random_state,
    )

    aucs = []
    losses = []

    # Logistic regression inside a standardization pipeline.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        ),
    )

    for train_idx, test_idx in cv.split(X, Y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # In rare folds, one class may disappear. Skip those folds.
        if np.unique(y_train).size < 2:
            continue

        clf.fit(X_train, y_train)
        # Probability of the active state
        proba = clf.predict_proba(X_test)[:, 1]

        try:
            auc_val = roc_auc_score(y_test, proba)
        except ValueError:
            # If only one class in y_test, AUC is undefined; skip this fold.
            continue

        try:
            loss_val = log_loss(y_test, proba, labels=[0, 1])
        except ValueError:
            # log_loss can occasionally fail for degenerate cases; skip.
            continue

        if np.isfinite(auc_val):
            aucs.append(float(auc_val))
        if np.isfinite(loss_val):
            losses.append(float(loss_val))

    if not aucs:
        # No valid folds; return ID only.
        return str(res_key), {
            "id": float(id_est),
            "state_score": float("nan"),
            "score_type": metric.upper(),
            "auc": float("nan"),
            "cross_entropy": float("nan"),
        }

    auc_mean = float(np.mean(aucs))
    ce_mean = float(np.mean(losses)) if losses else float("nan")

    # ------------------------------------------------------------------
    # Normalized state score
    # ------------------------------------------------------------------
    metric = metric.lower()
    score_type = "AUC" if metric == "auc" else "CE"

    if metric == "ce" and np.isfinite(ce_mean):
        # Cross-entropy → normalized information gain in [0, 1].
        # Baseline loss is the entropy H(Y) of the label distribution.
        p_active = float(Y.mean())
        eps = 1e-7
        p_active = min(max(p_active, eps), 1.0 - eps)
        baseline_ce = -(
            p_active * np.log(p_active) + (1.0 - p_active) * np.log(1.0 - p_active)
        )

        if baseline_ce > 0.0:
            info_gain = max(0.0, baseline_ce - ce_mean)
            state_score = min(1.0, info_gain / baseline_ce)
            raw_score = info_gain
        else:
            state_score = 0.0
            raw_score = 0.0
    else:
        # Default: AUC-based score, normalized to [0, 1]:
        #     AUC = 0.5 → 0 (random)
        #     AUC = 1.0 → 1 (perfect)
        auc_clipped = min(max(auc_mean, 0.0), 1.0)
        state_score = max(0.0, min(1.0, 2.0 * (auc_clipped - 0.5)))
        raw_score = auc_mean

    metrics = {
        "id": float(id_est),
        "state_score": float(state_score),
        "score_raw": float(raw_score),
        "score_type": score_type,
        "auc": float(auc_mean),
        "cross_entropy": float(ce_mean),
    }
    return str(res_key), metrics


class StaticStateSensitivity(BaseStaticReporter):
    """Intrinsic Dimension + State Sensitivity (classifier-based).

    This component replaces the earlier MI-based static analysis. It computes,
    for each residue, a pair of metrics:

        * ``id``          – intrinsic dimension via PCA variance threshold.
        * ``state_score`` – normalized classifier-based state sensitivity.

    The state score is, by default, based on cross-validated logistic-regression
    AUC; users can optionally choose a cross-entropy-based score by passing
    ``state_metric="ce"`` in the parameter dictionary.

    The returned dict is consumed by downstream analysis and visualization layers.
    """

    def _get_worker_function(self) -> Callable:
        return _static_worker_state

    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        # State metric: "auc" (default) or "ce".
        state_metric = str(kwargs.get("state_metric", "auc")).lower()
        if state_metric not in {"auc", "ce"}:
            state_metric = "auc"

        n_splits = int(kwargs.get("cv_splits", 5))
        random_state = int(kwargs.get("random_state", 0))
        id_var_thr = float(kwargs.get("id_variance_threshold", 0.9))

        return dict(
            metric=state_metric,
            n_splits=n_splits,
            random_state=random_state,
            id_variance_threshold=id_var_thr,
        )
