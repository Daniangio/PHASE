"""
Goal 2: The Static Atlas (Set Cover / Dominating Set).

This version builds a pooled Information-Imbalance atlas (single feature set)
and biases selection with state-sensitivity scores from Goal 1. Coverage is
soft (no hard Δ threshold) and explicitly rewarded in the QUBO via a
facility-location-style surrogate plus redundancy penalties.

Expected input from runner:
    analyzer = QUBOMaxCoverage()
    res = analyzer.run(
        features,
        candidate_indices=...,
        candidate_state_scores=...,
        **params,
    )

Where:
    features : FeatureDict
        Mapping residue index -> np.ndarray of shape (n_frames, d) built from
        the pooled simulations.
    candidate_indices : List[int] or List[str]
        Indices of residues considered in the QUBO.
    candidate_state_scores : Dict[index, float]
        State sensitivity score (e.g. JSD) from Goal 1, per residue.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import multiprocessing as mp
from typing import Optional, Tuple, List, Dict, Any, Sequence, Hashable, Iterable

from alloskin.common.types import FeatureDict

# Optional QUBO stack
try:
    import pyqubo
    from neal import SimulatedAnnealingSampler
    QUBO_AVAILABLE = True
except ImportError:
    QUBO_AVAILABLE = False

# Optional dadapy dependency for Information Imbalance
try:
    from dadapy.metric_comparisons import MetricComparisons
    DADAPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DADAPY_AVAILABLE = False


_IMBALANCE_SHARED = {
    "keys_list": None,
    "coords": None,
    "ranks": None,
    "missing": None,
}


def _init_imbalance_worker(keys_list, coords, ranks, missing_map=None):
    """Store shared data in globals for pool workers to avoid large pickling."""
    _IMBALANCE_SHARED["keys_list"] = keys_list
    _IMBALANCE_SHARED["coords"] = coords
    _IMBALANCE_SHARED["ranks"] = ranks
    _IMBALANCE_SHARED["missing"] = missing_map


def _compute_imbalance_row(task_args):
    """Worker to compute imbalance entries for a single source residue."""
    a, k_i, mc_i = task_args
    keys_list = _IMBALANCE_SHARED["keys_list"]
    coords = _IMBALANCE_SHARED["coords"]
    ranks = _IMBALANCE_SHARED["ranks"]
    missing_map = _IMBALANCE_SHARED.get("missing")

    dim_indices = list(range(coords[k_i].shape[1]))

    cols = None
    if missing_map is not None:
        cols = missing_map.get(a)

    row_results = []
    targets = cols if cols is not None else range(a + 1, len(keys_list))
    for b in targets:
        k_j = keys_list[b]
        ranks_j = ranks[k_j]

        try:
            imb_ji, imb_ij = mc_i.return_inf_imb_target_selected_coords(
                target_ranks=ranks_j,
                coord_list=[dim_indices],
            )
        except Exception:
            # treat as fully imbalanced (no coverage)
            imb_ij = 1.0
            imb_ji = 1.0

        row_results.append((b, float(imb_ij), float(imb_ji)))

    return a, row_results


class QUBOMaxCoverage:
    """
    Static Atlas via QUBO:
    ----------------------

    Let x_i be a binary variable indicating whether residue i is in the Basis Set.

    Definitions (pooled features):

        w_ij  : soft coverage weight of child j by parent i,
                derived from pooled Information Imbalance Δ(i→j).
        hub_i = sum_j w_ij  (hub score)

    Objective (to minimize):

        H =  sum_i [ + alpha * x_i
                     - beta_switch   * s_i     * x_i
                     - beta_hub      * hub_i   * x_i ]
             + sum_{i<j} [ beta_coverage   * cov_overlap_ij * x_i * x_j
                           + gamma_redundancy * overlap_ij  * x_i * x_j ]

    where:
        s_i              : max-normalized state-sensitivity score.
        cov_overlap_ij   : product overlap sum_k w_ik * w_jk (Noisy-OR surrogate
                           for the facility-location max).
        overlap_ij       : normalized weighted overlap of coverage domains
                           (sum_k min(w_ik, w_jk) / min(hub_i, hub_j)).

    Soft coverage weights:

        w_ij = max(0, 1 - (Δ(i→j)/ii_scale)) ** p

    With ii_scale ≈ 0.6, any Δ ≥ 0.6 yields w_ij = 0 → no coverage, no hub.
    """

    def __init__(self) -> None:
        # Used to store imbalance matrices on last run (pooled features)
        self._IIM_list: List[np.ndarray] = []

    @staticmethod
    def _normalize_key(key: Hashable) -> Hashable:
        """Coerce residue identifiers (e.g., '123' vs 123) to a stable type."""
        if isinstance(key, str):
            try:
                return int(key)
            except (TypeError, ValueError):
                return key
        if isinstance(key, (np.integer,)):
            return int(key)
        return key

    def _normalize_feature_keys(self, features: FeatureDict) -> Dict[Hashable, np.ndarray]:
        """
        Normalize feature keys to avoid silent drops from str/int mismatches.

        Raises if two distinct keys collapse to the same normalized key.
        """
        normalized: Dict[Hashable, np.ndarray] = {}
        reverse: Dict[Hashable, Hashable] = {}

        for key, val in features.items():
            norm_key = self._normalize_key(key)
            if norm_key in normalized and reverse[norm_key] != key:
                raise ValueError(
                    f"Feature keys {reverse[norm_key]} and {key} both map to normalized id {norm_key}."
                )
            normalized[norm_key] = val
            reverse[norm_key] = key

        return normalized

    def _normalize_mapping_keys(self, mapping: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Normalize arbitrary mapping keys (e.g., static results, state scores).
        """
        normalized: Dict[Hashable, Any] = {}
        reverse: Dict[Hashable, Hashable] = {}

        for key, val in mapping.items():
            norm_key = self._normalize_key(key)
            if norm_key in normalized and reverse[norm_key] != key:
                raise ValueError(
                    f"Keys {reverse[norm_key]} and {key} both map to normalized id {norm_key}."
                )
            normalized[norm_key] = val
            reverse[norm_key] = key

        return normalized

    # ------------------------------------------------------------------
    # Imbalance cache helpers
    # ------------------------------------------------------------------
    def _load_imbalance_cache_file(self, path: Path) -> Dict[str, Any] | None:
        """
        Load an imbalance cache file. Supports npz with 'imbalance_matrix' and 'keys'
        or legacy npy (assumed aligned to caller-provided keys).
        """
        if not path.exists() or not path.is_file():
            return None
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                mat = np.asarray(data["imbalance_matrix"], dtype=float)
                keys_arr = data.get("keys")
                keys_list = list(keys_arr.tolist()) if keys_arr is not None else None
                return {"matrix": mat, "keys": keys_list, "source": str(path)}
            elif isinstance(data, np.ndarray):
                return {"matrix": np.asarray(data, dtype=float), "keys": None, "source": str(path)}
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[QUBO] Warning: failed to load imbalance cache {path}: {exc}")
        return None

    def _prefill_from_cache(
        self,
        keys: Sequence[Hashable],
        cache_entries: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray | None, Dict[str, Any]]:
        """
        Merge cached imbalance values onto the candidate key set.

        Returns the prefill matrix (nan where missing) and metadata on reuse.
        """
        if not cache_entries:
            return None, {"cached_pairs": 0, "sources": []}

        keys_list = list(keys)
        key_to_idx = {k: i for i, k in enumerate(keys_list)}
        n = len(keys_list)
        prefill = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(prefill, 0.0)

        cached_pairs = 0
        sources = []

        for entry in cache_entries:
            mat = np.asarray(entry.get("matrix"), dtype=float)
            entry_keys = entry.get("keys")
            source = entry.get("source")
            if entry_keys is None:
                # Legacy: assume caller alignment if shape matches.
                if mat.shape != (n, n):
                    continue
                entry_keys = keys_list
            if len(entry_keys) != mat.shape[0] or mat.shape[0] != mat.shape[1]:
                continue

            entry_keys = [self._normalize_key(k) for k in entry_keys]
            sources.append(source)

            for i, ki in enumerate(entry_keys):
                ti = key_to_idx.get(ki)
                if ti is None:
                    continue
                row = mat[i]
                for j, kj in enumerate(entry_keys):
                    tj = key_to_idx.get(kj)
                    if tj is None:
                        continue
                    val = row[j]
                    if np.isnan(prefill[ti, tj]):
                        prefill[ti, tj] = float(val)
                        cached_pairs += 1

        return prefill, {"cached_pairs": cached_pairs, "sources": sources}

    def _build_union_cache(
        self,
        current_keys: Sequence[Hashable],
        current_matrix: np.ndarray,
        cache_entries: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Hashable], np.ndarray]:
        """
        Build a union cache over previous entries and the current matrix.
        """
        union_keys: List[Hashable] = []

        def _add_keys(seq: Iterable[Hashable]):
            for k in seq:
                if k not in union_keys:
                    union_keys.append(k)

        for entry in cache_entries:
            entry_keys = entry.get("keys") or []
            entry_keys = [self._normalize_key(k) for k in entry_keys]
            _add_keys(entry_keys)
        _add_keys(current_keys)

        n_union = len(union_keys)
        union_mat = np.full((n_union, n_union), np.nan, dtype=float)
        np.fill_diagonal(union_mat, 0.0)

        def _fill_from(keys_src: Sequence[Hashable], mat_src: np.ndarray):
            if mat_src.shape[0] != mat_src.shape[1]:
                return
            if len(keys_src) != mat_src.shape[0]:
                return
            idx_map = {k: i for i, k in enumerate(union_keys)}
            for i, ki in enumerate(keys_src):
                ui = idx_map.get(self._normalize_key(ki))
                if ui is None:
                    continue
                for j, kj in enumerate(keys_src):
                    uj = idx_map.get(self._normalize_key(kj))
                    if uj is None:
                        continue
                    if np.isnan(union_mat[ui, uj]):
                        union_mat[ui, uj] = float(mat_src[i, j])

        for entry in cache_entries:
            entry_keys = entry.get("keys")
            if entry_keys is None:
                continue
            entry_keys = [self._normalize_key(k) for k in entry_keys]
            _fill_from(entry_keys, np.asarray(entry.get("matrix"), dtype=float))

        _fill_from(current_keys, current_matrix)

        return union_keys, union_mat

    def _save_imbalance_cache(
        self,
        path: Path,
        keys: Sequence[Hashable],
        matrix: np.ndarray,
    ) -> None:
        """Persist imbalance cache with keys."""
        try:
            np.savez_compressed(
                path,
                imbalance_matrix=matrix,
                keys=np.array(list(keys), dtype=object),
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[QUBO] Warning: failed to save imbalance cache to {path}: {exc}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        features: FeatureDict,
        *,
        candidate_indices: Sequence[Hashable] | None = None,
        candidate_state_scores: Dict[Hashable, float] | None = None,
        static_results: Dict[str, Any] | None = None,
        static_results_path: str | None = None,
        filter_min_id: float = 1.5,
        filter_top_jsd: int | None = 20,
        filter_top_total: int | None = 120,
        # Imbalance reuse / persistence
        imbalance_matrix: np.ndarray | None = None,
        imbalance_matrix_path: str | Path | None = None,
        imbalance_matrix_paths: Sequence[str | Path] | None = None,
        imbalance_entries: Sequence[Dict[str, Any]] | None = None,
        save_imbalance_path: str | Path | None = None,
        # Info imbalance / coverage hyperparameters
        ii_scale: float = 0.6,
        soft_threshold_power: float = 2.0,
        # State-score regularization
        ii_threshold: float | None = 0.9,
        maxk: int | None = None,
        # QUBO hyperparameters
        alpha: float = 1.0,
        beta_switch: float = 5.0,
        beta_hub: float = 1.0,
        beta_coverage: float = 1.0,
        gamma_redundancy: float = 2.0,
        # Solver
        num_solutions: int = 5,
        num_reads: int = 2000,
        seed: int | None = None,
        # Logging
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        features
            FeatureDict mapping residue index -> array of shape (n_frames, d).
            Represents the combined (pooled) feature set to optimize over.
            Residue identifiers are normalized (e.g., "123" → 123) to avoid
            silent mismatches.
        candidate_indices
            Residues to be considered in the QUBO.
        candidate_state_scores
            Mapping residue index -> state sensitivity score (e.g. JSD).
            If None, all scores are treated as 0.
        static_results
            Optional static analysis results used to pre-filter candidates and
            populate state scores when candidate_indices is not provided.
            static_results_path
            Path to a JSON file with static analysis results; used only when
            static_results is not provided.
        filter_min_id
            Minimum intrinsic-dimension threshold for inclusion when filtering
            from static_results/static_results_path.
        filter_top_jsd
            Guaranteed number of top state-score residues to keep when
            filtering from static results (None keeps all passing filter_min_id).
        filter_top_total
            Cap on total residues kept after filling by intrinsic dimension
            (None keeps all passing filter_min_id).
        imbalance_matrix
            Optional precomputed imbalance matrix (aligned to `candidate_indices`).
        imbalance_matrix_path
            If provided and exists, load imbalance matrix from this path (legacy).
            Shape must match (N, N) for the candidate set unless keys are stored.
        imbalance_matrix_paths
            Optional list of cache paths (npz preferred) that include matrix + keys.
        imbalance_entries
            Optional list of explicit cache entries of the form
            {"matrix": ndarray, "keys": List[Hashable]}.
        save_imbalance_path
            If provided, save/merge imbalance data (matrix + residue keys) to this path
            to reuse in future runs.
        ii_scale
            Scale / threshold parameter for normalizing Δ before soft
            coverage. If Δ >= ii_scale → w_ij = 0 (no coverage, no hub).
        soft_threshold_power
            Exponent p in w_ij = max(0, 1 - Δ/ii_scale)^p. Higher p sharpens
            the distinction between strong and weak coverage.
        ii_threshold
            Optional lower-bound for the max-normalization denominator. Scores
            are mapped to s_i = score / max(max_score, ii_threshold) ∈ [0, 1].
        maxk
            Maximum neighborhood size for dadapy MetricComparisons
            (defaults to n_samples - 1).
        alpha
            Baseline linear cost per selected residue.
        beta_switch
            Linear reward for state sensitivity (switch-like behavior).
        beta_hub
            Linear reward for hub score (total coverage surrogate).
        beta_coverage
            Pairwise penalty weight from the Noisy-OR coverage surrogate:
            rewards union coverage by penalizing product overlaps.
        gamma_redundancy
            Quadratic penalty for normalized redundancy (mutual coverage).
        num_solutions
            Number of unique solutions to report from the annealer.
        num_reads
            Number of SA reads.
        seed
            Optional RNG seed for deterministic sampling.

        Returns
        -------
        Dict with fields:
            - "solutions": list of solution dicts
            - "matrix_indices": ordered list of residue indices used in QUBO
            - "imbalance_matrix": Δ_avg(i→j) as nested list
            - "coverage_weights": w_ij (avg) as nested list
            - "hub_scores": hub_i (avg) per residue index
            - "regularized_state_scores": max-normalized s_i
            - "parameters": hyperparameters used
            - "imbalance_cache": metadata about load/save paths
            - "error": only present if QUBO stack not available or failure
        """
        if not QUBO_AVAILABLE:
            return {"error": "pyqubo / neal not available; cannot run QUBO."}
        
        # ------------------------------------------------------------
        # Helpers for printing / callback
        # ------------------------------------------------------------
        def report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)
            else:
                print(f"[{pct}%] {msg}")

        features = self._normalize_feature_keys(features)

        # Determine candidate pool and accompanying state scores.
        keys, candidate_state_scores = self._prepare_candidates(
            features=features,
            candidate_indices=candidate_indices,
            candidate_state_scores=candidate_state_scores,
            static_results=static_results,
            static_results_path=static_results_path,
            filter_min_id=filter_min_id,
            filter_top_jsd=filter_top_jsd,
            filter_top_total=filter_top_total,
        )

        if len(keys) == 0:
            return {"error": "No candidate indices provided to QUBOMaxCoverage."}

        raw_state_scores = {k: float(candidate_state_scores.get(k, 0.0)) for k in keys}

        # --------------------------------------------------------------
        # 0. Regularize state scores with max-normalization (optional floor)
        # --------------------------------------------------------------
        max_score = max(raw_state_scores.values()) if raw_state_scores else 0.0
        denom_candidates = [max_score]
        if ii_threshold is not None and ii_threshold > 0.0:
            denom_candidates.append(ii_threshold)
        denom = max(denom_candidates) if denom_candidates else 1.0

        if denom <= 0.0:
            candidate_state_scores = {k: 0.0 for k in keys}
        else:
            candidate_state_scores = {
                k: min(1.0, raw_state_scores.get(k, 0.0) / denom) for k in keys
            }

        # --------------------------------------------------------------
        # 1. Build feature arrays for each candidate
        # --------------------------------------------------------------
        features_act = features
        feat_act: Dict[Hashable, np.ndarray] = {}

        for k in keys:
            if k not in features_act:
                raise ValueError(f"Residue key {k} is missing in provided features.")
            arr_act = np.asarray(features_act[k])

            # Coerce features to 2D (n_frames, d)
            if arr_act.ndim == 1:
                arr_act = arr_act.reshape(-1, 1)
            elif arr_act.ndim > 2:
                arr_act = arr_act.reshape(arr_act.shape[0], -1)

            if arr_act.ndim != 2:
                raise ValueError(f"Feature arrays for key {k} must be 2D (n_frames, d).")

            feat_act[k] = arr_act

        # --------------------------------------------------------------
        # 2. Compute or load Information Imbalance Δ(i→j)
        # --------------------------------------------------------------
        cache_entries: List[Dict[str, Any]] = []
        cache_sources: List[str] = []

        if imbalance_entries:
            cache_entries.extend(list(imbalance_entries))

        if imbalance_matrix is not None:
            cache_entries.append({"matrix": np.asarray(imbalance_matrix, dtype=float), "keys": list(keys)})

        combined_paths: List[Path] = []
        if imbalance_matrix_paths:
            combined_paths.extend([Path(p) for p in imbalance_matrix_paths])
        if imbalance_matrix_path:
            combined_paths.append(Path(imbalance_matrix_path))

        for p in combined_paths:
            entry = self._load_imbalance_cache_file(p)
            if entry:
                cache_entries.append(entry)
                if entry.get("source"):
                    cache_sources.append(entry["source"])

        prefill, cache_meta = self._prefill_from_cache(keys, cache_entries)
        cached_pairs = cache_meta.get("cached_pairs", 0)
        cache_sources = cache_sources or cache_meta.get("sources", [])

        if prefill is not None:
            missing_pairs = int(np.isnan(prefill).sum())
        else:
            missing_pairs = len(keys) * len(keys) - len(keys)

        print(f"[QUBO] Computing Information Imbalance for {len(keys)} candidates... "
              f"(cached pairs reused: {cached_pairs})")

        imbalance_matrix = self._compute_imbalance_matrix(
            keys,
            feat_act,
            maxk=maxk,
            prefill=prefill,
        )

        self._IIM_list.append(imbalance_matrix)

        # Update union cache for persistence
        union_size = None
        if save_imbalance_path:
            union_keys, union_mat = self._build_union_cache(keys, imbalance_matrix, cache_entries)
            union_size = len(union_keys)
            self._save_imbalance_cache(Path(save_imbalance_path), union_keys, union_mat)

        # --------------------------------------------------------------
        # 3. Convert Δ to soft coverage weights w_ij (avg)
        # --------------------------------------------------------------
        coverage_weights, hub_scores = self._compute_soft_coverage(
            keys,
            imbalance_matrix,
            ii_scale=ii_scale,
            power=soft_threshold_power,
        )

        # --------------------------------------------------------------
        # 4. Build QUBO (h_i, J_ij) using average hub / coverage
        # --------------------------------------------------------------
        h_linear, J_quadratic = self._build_hamiltonian(
            keys,
            coverage_weights,
            hub_scores,
            candidate_state_scores,
            alpha=alpha,
            beta_switch=beta_switch,
            beta_hub=beta_hub,
            beta_coverage=beta_coverage,
            gamma_redundancy=gamma_redundancy,
        )

        # --------------------------------------------------------------
        # 5. Solve QUBO via Simulated Annealing
        # --------------------------------------------------------------
        try:
            x_vars = {str(k): pyqubo.Binary(str(k)) for k in keys}
            H_expr = 0.0

            for k, val in h_linear.items():
                H_expr += val * x_vars[str(k)]

            for (ki, kj), val in J_quadratic.items():
                if val == 0.0:
                    continue
                H_expr += val * x_vars[str(ki)] * x_vars[str(kj)]

            model = H_expr.compile()
            qubo, offset = model.to_qubo()

            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(
                qubo,
                num_reads=num_reads,
                seed=seed,
            )

            # Collect unique solutions (by binary pattern)
            solutions: List[Dict[str, Any]] = []
            seen_patterns = set()

            for sample, energy in zip(sampleset.record.sample, sampleset.record.energy):
                pattern = tuple(int(v) for v in sample)
                if pattern in seen_patterns:
                    continue
                seen_patterns.add(pattern)

                selected_indices = [
                    keys[i] for i, bit in enumerate(pattern) if bit == 1
                ]

                # Compute union coverage and per-residue metrics for this solution
                union_coverage, per_parent_coverage = \
                    self._compute_union_coverage(keys, coverage_weights, pattern)

                solutions.append(
                    {
                        "selected": selected_indices,
                        "energy": float(energy + offset),
                        "raw_energy": float(energy),
                        "union_coverage": float(union_coverage),
                        "per_parent_coverage": per_parent_coverage,
                        "pattern": pattern,
                    }
                )

                if len(solutions) >= num_solutions:
                    break

            result: Dict[str, Any] = {
                "solutions": solutions,
                "matrix_indices": keys,
                "imbalance_matrix": imbalance_matrix.tolist(),
                "coverage_weights": coverage_weights.tolist(),
                "hub_scores": {k: float(hub_scores[i]) for i, k in enumerate(keys)},
                "raw_state_scores": {k: float(raw_state_scores.get(k, 0.0)) for k in keys},
                "regularized_state_scores": {
                    k: float(candidate_state_scores[k]) for k in keys
                },
                "parameters": {
                    "alpha": alpha,
                    "beta_switch": beta_switch,
                    "beta_hub": beta_hub,
                    "beta_coverage": beta_coverage,
                    "gamma_redundancy": gamma_redundancy,
                    "ii_scale": ii_scale,
                    "ii_threshold": ii_threshold,
                    "soft_threshold_power": soft_threshold_power,
                    "maxk": maxk,
                    "num_reads": num_reads,
                    "num_solutions": num_solutions,
                },
            "imbalance_cache": {
                "cached_pairs": int(cached_pairs),
                "missing_pairs_after_cache": int(missing_pairs),
                "sources": list(cache_sources),
                "save_path": str(save_imbalance_path) if save_imbalance_path else None,
                "loaded_paths": [str(p) for p in combined_paths],
                "union_size": int(union_size) if union_size is not None else None,
            },
        }

            return result

        except Exception as e:  # pragma: no cover
            print(f"[QUBO] Error solving QUBO: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_static_results(
        self, data: Dict[str, Any] | None
    ) -> Dict[str, Dict[str, Any]] | None:
        """
        Extract a residue->stats mapping from a user-provided object.

        Accepts either the raw static results dict or a wrapper dict that
        contains such a mapping.
        """
        if not isinstance(data, dict):
            return None

        dict_values = {k: v for k, v in data.items() if isinstance(v, dict)}
        if dict_values and any("state_score" in v for v in dict_values.values()):
            return dict_values

        # Try to find a nested mapping
        for val in data.values():
            if isinstance(val, dict):
                nested = self._normalize_static_results(val)
                if nested:
                    return nested
        return None

    def _prepare_candidates(
        self,
        *,
        features: FeatureDict,
        candidate_indices: Sequence[Hashable] | None,
        candidate_state_scores: Dict[Hashable, float] | None,
        static_results: Dict[str, Any] | None,
        static_results_path: str | None,
        filter_min_id: float,
        filter_top_jsd: int | None,
        filter_top_total: int | None,
    ) -> Tuple[List[Hashable], Dict[Hashable, float]]:
        """
        Determine candidate residues and their state scores.

        Priority:
          1) static_results/static_results_path → apply filtering
          2) explicit candidate_indices
          3) fallback to all residues in `features`

        Residue identifiers are normalized (e.g., "123" → 123) to avoid
        silent drops from type mismatches.
        """
        if candidate_state_scores is not None:
            candidate_state_scores = self._normalize_mapping_keys(candidate_state_scores)
        else:
            candidate_state_scores = {}

        static_data = static_results
        if static_data is None and static_results_path:
            try:
                with open(static_results_path, "r") as fh:
                    static_data = json.load(fh)
            except Exception as exc:
                raise ValueError(
                    f"Failed to load static_results_path '{static_results_path}': {exc}"
                )

        parsed_static = self._normalize_static_results(static_data) if static_data else None
        parsed_static = (
            self._normalize_mapping_keys(parsed_static) if parsed_static is not None else None
        )

        if parsed_static:
            movers = []
            for key, stats in parsed_static.items():
                if not isinstance(stats, dict):
                    continue
                id_val = float(stats.get("id", 0.0) or 0.0)
                state_score = float(stats.get("state_score", 0.0) or 0.0)
                if id_val >= filter_min_id:
                    movers.append((key, id_val, state_score))

            if filter_top_jsd is None:
                filter_top_jsd = len(movers)
            if filter_top_total is None:
                filter_top_total = len(movers)

            by_jsd = sorted(movers, key=lambda x: x[2], reverse=True)
            selected: List[Hashable] = []
            for key, _, _ in by_jsd:
                if len(selected) >= filter_top_jsd:
                    break
                selected.append(key)

            by_entropy = sorted(movers, key=lambda x: x[1], reverse=True)
            for key, _, _ in by_entropy:
                if len(selected) >= filter_top_total:
                    break
                if key not in selected:
                    selected.append(key)

            keys = [k for k in selected if k in features]
            state_scores = {
                k: float(parsed_static.get(k, {}).get("state_score", 0.0) or 0.0)
                for k in keys
            }
            return keys, state_scores

        # No static filtering → use user-provided candidates or everything
        if candidate_indices is not None:
            keys = []
            for k in candidate_indices:
                norm_k = self._normalize_key(k)
                if norm_k in features:
                    keys.append(norm_k)
        else:
            keys = list(features.keys())

        base_scores = candidate_state_scores
        state_scores = {k: float(base_scores.get(k, 0.0)) for k in keys}
        return keys, state_scores

    def _compute_imbalance_matrix(
        self,
        keys: Sequence[Hashable],
        feat_act: Dict[Hashable, np.ndarray],
        *,
        maxk: int | None,
        prefill: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute Information Imbalance Δ(i→j) using dadapy, averaging
        both directions to form a symmetric matrix.
        """
        if not DADAPY_AVAILABLE:
            raise ImportError("dadapy is required for imbalance computation.")

        n = len(keys)
        keys_list = list(keys)

        # Prepare MetricComparisons and ranks for each residue
        n_samples = next(iter(feat_act.values())).shape[0]
        maxk = n_samples - 1 if maxk is None else min(maxk, n_samples - 1)

        coords: Dict[Hashable, np.ndarray] = {}
        comps: Dict[Hashable, MetricComparisons] = {}
        ranks: Dict[Hashable, np.ndarray] = {}

        half_period = np.pi
        for key in keys_list:
            arr = np.clip(feat_act[key].reshape(n_samples, -1) + half_period, 0, 2*half_period)
            coords[key] = arr
            mc = MetricComparisons(coordinates=arr, maxk=maxk, n_jobs=1)
            mc.compute_distances(period=2*half_period)
            comps[key] = mc
            ranks[key] = mc.dist_indices

        # Initialize with prefill if provided (nan indicates missing)
        if prefill is not None:
            IIM = np.asarray(prefill, dtype=float).copy()
        else:
            IIM = np.full((n, n), np.nan, dtype=float)

        np.fill_diagonal(IIM, 0.0)

        # Determine missing pairs
        missing_map: Dict[int, List[int]] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if np.isnan(IIM[i, j]) or np.isnan(IIM[j, i]):
                    missing_map.setdefault(i, []).append(j)

        if not missing_map:
            return IIM

        # Parallelize imbalance computation row-wise; each worker receives
        # one MetricComparisons instance (mc_i) and computes missing k_js.
        tasks = [(a, keys_list[a], comps[keys_list[a]]) for a in missing_map.keys()]
        cpu_total = mp.cpu_count() or 1
        num_workers = min(len(tasks), max(1, cpu_total))

        try:
            ctx = mp.get_context("fork")
        except (AttributeError, ValueError):
            ctx = mp.get_context() if hasattr(mp, "get_context") else mp

        try:
            if num_workers == 1:
                # Avoid Pool startup cost when only one worker is available.
                _init_imbalance_worker(keys_list, coords, ranks, missing_map)
                results = [_compute_imbalance_row(task) for task in tasks]
            else:
                with ctx.Pool(
                    processes=num_workers,
                    initializer=_init_imbalance_worker,
                    initargs=(keys_list, coords, ranks, missing_map),
                ) as pool:
                    results = pool.map(_compute_imbalance_row, tasks)
        except Exception:
            # Fallback to sequential computation if multiprocessing fails.
            _init_imbalance_worker(keys_list, coords, ranks, missing_map)
            results = [_compute_imbalance_row(task) for task in tasks]

        for a, row_results in results:
            for b, imb_ij, imb_ji in row_results:
                IIM[a, b] = imb_ij
                IIM[b, a] = imb_ji

        return IIM

    def _compute_soft_coverage(
        self,
        keys: Sequence[Hashable],
        imbalance_matrix: np.ndarray,
        *,
        ii_scale: float,
        power: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert imbalance Δ(i→j) into soft coverage weights w_ij:

            w_ij = max(0, 1 - Δ(i→j)/ii_scale) ** power

        With this choice, if Δ >= ii_scale → w_ij = 0
        (no coverage / no contribution to hub score).

        Returns
        -------
        coverage_weights : ndarray, shape (N, N)
        hub_scores       : ndarray, shape (N,)
        """
        N = len(keys)
        D = imbalance_matrix.copy()

        D_scaled = D / float(ii_scale)
        D_scaled = np.clip(D_scaled, 0.0, 1.0)

        W = np.power(np.maximum(0.0, 1.0 - D_scaled), power)
        np.fill_diagonal(W, 0.0)

        hub_scores = W.sum(axis=1)
        return W, hub_scores

    def _build_hamiltonian(
        self,
        keys,
        coverage_weights,
        hub_scores,
        state_scores,
        *,
        alpha,
        beta_switch,
        beta_hub,
        beta_coverage,
        gamma_redundancy,   # NEW unified redundancy term
    ):
        """
        Build QUBO combining a coverage surrogate with normalized redundancy.
        """

        N = len(keys)
        keys_list = list(keys)

        h_linear = {}
        J_quadratic = {}

        W = coverage_weights

        # ------------------------------
        # Linear coefficients
        # ------------------------------
        for idx_i, key_i in enumerate(keys_list):
            s_i = float(state_scores.get(key_i, 0.0))
            hub_i = float(hub_scores[idx_i])

            h = alpha
            h -= beta_switch * s_i
            h -= beta_hub * hub_i

            h_linear[key_i] = float(h)

        # ------------------------------
        # Quadratic overlap penalty
        # ------------------------------
        for i in range(N):
            for j in range(i + 1, N):

                # Coverage surrogate (Noisy-OR truncated at pairwise order):
                # penalize joint selections that cover the same children.
                coverage_overlap = float(np.dot(W[i], W[j]))

                # Normalized redundancy: penalize local overlap without
                # over-penalizing global hubs.
                raw_overlap = float(np.minimum(W[i], W[j]).sum())
                denom = max(min(hub_scores[i], hub_scores[j]), 1e-12)
                normalized_overlap = raw_overlap / denom if denom > 0 else 0.0

                coeff = 0.0
                if coverage_overlap > 0.0:
                    coeff += beta_coverage * coverage_overlap
                if normalized_overlap > 0.0:
                    coeff += gamma_redundancy * normalized_overlap

                if coeff > 0.0:
                    J_quadratic[(keys_list[i], keys_list[j])] = coeff

        return h_linear, J_quadratic


    def _compute_union_coverage(
        self,
        keys: Sequence[Hashable],
        coverage_weights: np.ndarray,
        pattern: Sequence[int],
    ) -> Tuple[float, Dict[Hashable, float]]:
        """
        Given a binary selection pattern, compute:

        - union_coverage: sum over children j of max_i w_ij
        - per_parent_coverage: hub score restricted to selected parents
        """
        keys_list = list(keys)
        W = coverage_weights

        selected_indices = [i for i, bit in enumerate(pattern) if bit == 1]
        per_parent: Dict[Hashable, float] = {}

        if not selected_indices:
            return 0.0, {}

        for i in selected_indices:
            k = keys_list[i]
            per_parent[k] = float(W[i].sum())

        union_weights = np.max(W[selected_indices, :], axis=0)
        union_coverage = float(union_weights.sum())

        return union_coverage, per_parent
