"""Cluster per-residue angles inside selected metastable states and persist as NPZ."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree
import MDAnalysis as mda
from dadapy import Data

from backend.services.descriptors import load_descriptor_npz
from backend.services.project_store import DescriptorState, ProjectStore, SystemMetadata


def _slug(value: str) -> str:
    """Create a filesystem/NPZ-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "metastable"


def _kmeans_sweep(
    samples: np.ndarray, max_k: int, random_state: int
) -> Tuple[np.ndarray, int, float]:
    """Run KMeans sweep to find best K by silhouette score."""
    n_samples = samples.shape[0]
    if n_samples == 0:
        return np.array([], dtype=np.int32), 0, 0.0
    
    # If we only have 1 point or max_k=1, we can't do silhouette
    upper_k = max(1, min(int(max_k), n_samples))
    
    if upper_k == 1:
        return np.zeros(n_samples, dtype=np.int32), 1, 0.0

    best_k = 1
    best_score = -np.inf

    for k in range(1, upper_k + 1):
        if k == 1:
            score = 0.0  # fallback
        else:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(samples)
            n_labels = len(np.unique(labels))
            # silhouette_score requires 2 <= n_labels <= n_samples - 1
            if n_labels < 2 or n_labels >= n_samples:
                score = -np.inf
            else:
                try:
                    score = silhouette_score(samples, labels)
                except ValueError:
                    # Handle tiny sample sets where sklearn refuses to score
                    score = -np.inf
        
        # Prefer higher K slightly if scores are very close? No, stick to raw score.
        if score > best_score:
            best_score = score
            best_k = k

    if best_k == 1:
        final_labels = np.zeros(samples.shape[0], dtype=np.int32)
    else:
        km = KMeans(n_clusters=best_k, n_init="auto", random_state=random_state)
        final_labels = km.fit_predict(samples).astype(np.int32)

    return final_labels, best_k, best_score


def _angles_to_embedding(samples: np.ndarray) -> np.ndarray:
    """Convert angle triplets to sin/cos embedding for periodic clustering."""
    angles = samples[:, :3]
    emb = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)


def _cluster_residue_samples(
    samples: np.ndarray,
    max_k: int,
    random_state: int,
    *,
    algorithm: str = "tomato",
    density_z: float | str | None = None,
    density_maxk: Optional[int] = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    hierarchical_n_clusters: Optional[int] = None,
    hierarchical_linkage: str = "ward",
    tomato_k: int = 15,
    tomato_tau: float | str = "auto",
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Cluster angles with periodic distance. Returns labels, cluster count, diagnostics."""
    if samples.size == 0:
        return np.array([], dtype=np.int32), 0, {}

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[1] < 3:
        raise ValueError("Residue samples must be (n_frames, >=3) shaped.")

    # Use sin/cos embedding to respect angular periodicity.
    emb = _angles_to_embedding(samples)

    algo = (algorithm or "tomato").lower()
    diagnostics: Dict[str, Any] = {}
    density_z_val: float | str = "auto"
    if density_z is not None and not (isinstance(density_z, str) and density_z.lower() == "auto"):
        try:
            density_z_val = float(density_z)
        except Exception:
            density_z_val = "auto"
    density_maxk_val: int = max(1, int(density_maxk)) if density_maxk is not None else 100

    # --- Standard KMeans ---
    if algo == "kmeans":
        labels, k, _ = _kmeans_sweep(emb, max_k, random_state)
        return labels, k, diagnostics

    # --- Hierarchical ---
    if algo == "hierarchical":
        n_clusters = hierarchical_n_clusters or max_k
        n_clusters = max(1, min(int(n_clusters), emb.shape[0]))
        linkage = (hierarchical_linkage or "ward").lower()
        if linkage not in {"ward", "complete", "average", "single"}:
            linkage = "ward"
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(emb).astype(np.int32)
        k = len(np.unique(labels))
        return labels, int(k), diagnostics

    # --- Density Peaks ---
    if algo == "density_peaks":
        n = emb.shape[0]
        if n == 1:
            return np.zeros(1, dtype=np.int32), 1, diagnostics

        # Use DADApy for density peak clustering (adaptive density peak).
        try:
            dp_maxk = max(1, min(density_maxk_val, n - 1))
            dp_data = Data(coordinates=emb, maxk=dp_maxk, verbose=False)
            dp_data.compute_distances()
            dp_data.compute_id_2NN()
            dp_data.compute_density_kstarNN()
            if density_z_val == "auto":
                dp_data.compute_clustering_ADP()
                diagnostics["density_peaks_Z"] = "auto"
            else:
                dp_data.compute_clustering_ADP(Z=float(density_z_val))
                diagnostics["density_peaks_Z"] = float(density_z_val)
            labels = np.asarray(dp_data.cluster_assignment, dtype=np.int32)
            k_final = int(dp_data.N_clusters) if hasattr(dp_data, "N_clusters") else int(
                len([c for c in np.unique(labels) if c >= 0])
            )
            diagnostics["density_peaks_method"] = "dadapy_adp"
            diagnostics["density_peaks_k"] = k_final
            diagnostics["density_peaks_maxk"] = dp_maxk
        except Exception as exc:
            raise ValueError(f"DADApy density peaks failed: {exc}") from exc
        
        final_labels = labels

    # --- ToMATo ---
    elif algo == "tomato":
        n = emb.shape[0]
        if n == 1:
            return np.zeros(1, dtype=np.int32), 1, diagnostics
        
        t_k_val = tomato_k if tomato_k is not None else 15
        k_nn = max(1, min(int(t_k_val), n - 1))
        
        tree = KDTree(emb)
        dists, idxs = tree.query(emb, k=k_nn + 1)
        knn_idxs = idxs[:, 1:]
        d_k = dists[:, -1] + 1e-8
        
        # --- FIXED DENSITY ESTIMATOR ---
        # Old (Unstable): rho = 1.0 / (d_k ** max(dim, 1))
        # New (Stable): Log-density. 
        # Since d_k is small for dense regions, -log(d_k) is large positive.
        # This prevents "super peaks" from dominating the persistence diagram.
        rho = -np.log(d_k)
        
        local_max = np.zeros(n, dtype=bool)
        for i in range(n):
            neigh = knn_idxs[i]
            best_rho = rho[i]
            is_max = True
            for j in neigh:
                if rho[j] > best_rho:
                    is_max = False
                    break
            local_max[i] = is_max

        order = sorted(range(n), key=lambda idx: (-rho[idx], idx))
        uf_parent = np.arange(n, dtype=int)
        root_density = rho.copy()
        processed = np.zeros(n, dtype=bool)
        transitions: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []

        def find(x: int) -> int:
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return ra
            if root_density[ra] >= root_density[rb]:
                uf_parent[rb] = ra
                return ra
            uf_parent[ra] = rb
            return rb
        
        auto_tau = isinstance(tomato_tau, str) and tomato_tau.lower() == "auto"
        
        # Compute tau from gaps in persistence values if user requested "auto".
        def _auto_tau_from_persistence(values: List[float], fallback: float = 0.5) -> float:
            finite_vals = [float(v) for v in values if np.isfinite(v)]
            if not finite_vals:
                return fallback
            vals = sorted(finite_vals, reverse=True)
            if len(vals) == 1:
                # Use half of the observed barrier to keep threshold in-range
                return max(fallback, vals[0] * 0.5)
            best_gap = -np.inf
            best_idx = 0
            # Look for the largest gap to separate "real" peaks from "noise"
            for i in range(len(vals) - 1):
                high, low = vals[i], vals[i + 1]
                gap = high - low
                if gap > best_gap:
                    best_gap = gap
                    best_idx = i
            high, low = vals[best_idx], vals[best_idx + 1]
            return low + (high - low) * 0.5

        tau = 0.5 if auto_tau else float(tomato_tau if tomato_tau is not None else 0.5)
        diagnostics["tau_mode"] = "auto" if auto_tau else "manual"

        # --- ToMATo Loop ---
        pending_transitions: List[Dict[str, Any]] = []
        for idx in order:
            if local_max[idx]:
                processed[idx] = True
                continue
            
            processed[idx] = True
            neigh = [j for j in knn_idxs[idx] if processed[j] and rho[j] > rho[idx]]
            
            if not neigh:
                # Disconnected point. We explicitly leave it as self-parent.
                continue

            roots = []
            seen_roots = set()
            for j in neigh:
                rj = find(j)
                if rj not in seen_roots:
                    seen_roots.add(rj)
                    roots.append(rj)
            
            if len(roots) == 1:
                union(idx, roots[0])
                continue

            roots_sorted = sorted(roots, key=lambda r: (-root_density[r], r))
            r_max = roots_sorted[0]
            
            for r_other in roots_sorted[1:]:
                persistence = float(root_density[r_other] - rho[idx])
                event = {
                    "peak_density": float(root_density[r_other]),
                    "persistence": persistence,
                    "saddle_density": float(rho[idx]),
                }
                events.append(event)
                transition_record = {
                    "root_a": int(r_max),
                    "root_b": int(r_other),
                    "saddle_index": int(idx),
                    "persistence": persistence,
                }
                if auto_tau:
                    pending_transitions.append(transition_record)
                else:
                    if persistence < tau:
                        union(r_other, r_max)
                    else:
                        transitions.append(transition_record)
            union(idx, r_max)

        # Resolve tau automatically from persistence gaps, then apply merges.
        if auto_tau:
            persistence_values = [float(ev.get("persistence", 0.0)) for ev in events]
            # Since density is now log-scale, the fallback must be sensible for log-space
            # 0.1 in log space corresponds to exp(0.1) ~ 1.1x density ratio.
            tau = _auto_tau_from_persistence(persistence_values, fallback=0.1)
            diagnostics["tau"] = tau
            diagnostics["tau_candidates"] = persistence_values
            for tr in pending_transitions:
                ra, rb = find(tr["root_a"]), find(tr["root_b"])
                if ra == rb:
                    continue
                if tr["persistence"] < tau:
                    union(ra, rb)
                else:
                    transitions.append(tr)
        else:
            diagnostics["tau"] = tau

        # --- 1. Persistence-based merging (Water-filling) to respect max_k ---
        t_k_max = max_clusters_per_residue if max_k is None else max_k
        active_roots = {find(i) for i in range(n)}
        
        if len(active_roots) > t_k_max:
            merge_candidates: Dict[Tuple[int, int], float] = {}
            for tr in transitions:
                ra = find(tr["root_a"])
                rb = find(tr["root_b"])
                if ra == rb: continue
                key = tuple(sorted((ra, rb)))
                pers = float(tr.get("persistence", 0.0))
                if key not in merge_candidates or pers < merge_candidates[key]:
                    merge_candidates[key] = pers
            
            forced_merges: List[Dict[str, Any]] = []
            for (ra, rb), pers in sorted(merge_candidates.items(), key=lambda kv: kv[1]):
                root_a, root_b = find(ra), find(rb)
                if root_a == root_b: continue
                active_roots = {find(i) for i in range(n)}
                if len(active_roots) <= t_k_max: break
                union(root_a, root_b)
                forced_merges.append({"roots": (int(root_a), int(root_b)), "persistence": pers})
            diagnostics["forced_merges"] = forced_merges

        # --- Final Label Assignment ---
        root_map: Dict[int, int] = {}
        labels = np.full(n, -1, dtype=np.int32)
        next_id = 0
        
        # Pre-pass: Identify population of roots
        root_counts: Dict[int, int] = {}
        for i in range(n):
            r = find(i)
            root_counts[r] = root_counts.get(r, 0) + 1
            
        for i in range(n):
            r = find(i)
            # If strictly disconnected (size 1), mark as -1 so KMeans can handle it
            if root_counts[r] == 1:
                labels[i] = -1
            else:
                if r not in root_map:
                    root_map[r] = next_id
                    next_id += 1
                labels[i] = root_map[r]

        diagnostics["persistence_events"] = events
        diagnostics["transitions"] = transitions
        
        final_labels = labels
        k_final = next_id

    # --- DBSCAN ---
    else:
        eps = float(dbscan_eps) if dbscan_eps is not None else 0.5
        min_s = int(dbscan_min_samples) if dbscan_min_samples is not None else 5
        min_s = max(1, min_s)
        db = DBSCAN(eps=eps, min_samples=min_s, metric="euclidean")
        labels = db.fit_predict(emb).astype(np.int32)
        
        # Remap standard DBSCAN labels (-1 is noise)
        unique_clusters = sorted([int(v) for v in np.unique(labels) if v != -1])
        mapping = {old: idx for idx, old in enumerate(unique_clusters)}
        remapped = np.array([mapping.get(int(v), -1) for v in labels], dtype=np.int32)
        diagnostics["dbscan_unique"] = unique_clusters
        
        final_labels = remapped
        k_final = len(unique_clusters)

    # =========================================================
    # COMMON POST-PROCESSING: CLUSTER UNASSIGNED (-1) VIA KMEANS
    # =========================================================
    
    noise_mask = final_labels == -1
    n_noise = np.sum(noise_mask)
    
    if n_noise > 0:
        # Determine how many slots we have left
        available_slots = max(1, max_k - k_final)
        
        # Extract noise samples
        noise_samples = emb[noise_mask]
        
        # Run KMeans sweep on noise
        # Note: best_k_noise is 1-based count
        noise_labels, best_k_noise, score = _kmeans_sweep(
            noise_samples, 
            max_k=available_slots, 
            random_state=random_state
        )
        
        # If noise_labels returns empty (shouldn't happen given check), skip
        if noise_labels.size > 0:
            # Shift noise labels to start after existing clusters
            shifted_labels = noise_labels + k_final
            final_labels[noise_mask] = shifted_labels
            
            # Update total K
            k_final += best_k_noise
            
            diagnostics["noise_kmeans_k"] = int(best_k_noise)
            diagnostics["noise_kmeans_score"] = float(score)

    return final_labels, int(k_final), diagnostics


def _uniform_subsample_indices(n_frames: int, max_frames: int) -> np.ndarray:
    """Pick roughly uniform indices up to max_frames."""
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=int)
    idx = np.linspace(0, n_frames - 1, num=max_frames, dtype=int)
    return np.unique(idx)


def _assign_labels_by_knn(
    emb_full: np.ndarray,
    emb_subset: np.ndarray,
    subset_labels: np.ndarray,
    subset_indices: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:
    """Assign labels to remaining frames by majority vote of nearest neighbors."""
    n_frames = emb_full.shape[0]
    full_labels = np.full(n_frames, -1, dtype=np.int32)
    full_labels[subset_indices] = subset_labels.astype(np.int32)
    if emb_subset.shape[0] == 0 or k_neighbors <= 0:
        return full_labels

    k_neighbors = min(int(k_neighbors), emb_subset.shape[0])
    tree = KDTree(emb_subset)
    _, idxs = tree.query(emb_full, k=k_neighbors)
    if idxs.ndim == 1:
        idxs = idxs[:, None]

    for i in range(n_frames):
        if full_labels[i] != -1:
            continue
        neigh_labels = subset_labels[idxs[i]]
        valid = neigh_labels[neigh_labels >= 0]
        if valid.size == 0:
            continue
        vals, counts = np.unique(valid, return_counts=True)
        full_labels[i] = int(vals[np.argmax(counts)])

    return full_labels


def _cluster_with_subsample(
    samples: np.ndarray,
    max_k: int,
    random_state: int,
    *,
    algorithm: str,
    density_maxk: int,
    density_z: float | str | None,
    dbscan_eps: float,
    dbscan_min_samples: int,
    hierarchical_n_clusters: Optional[int],
    hierarchical_linkage: str,
    tomato_k: int,
    tomato_tau: float | str,
    max_cluster_frames: Optional[int],
    assign_k: int = 10,
    subsample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any], int]:
    """Cluster on a subsample if requested, then assign remaining frames by neighbor majority."""
    n_frames = samples.shape[0]
    if not max_cluster_frames or max_cluster_frames <= 0 or n_frames <= max_cluster_frames:
        labels, k, diag = _cluster_residue_samples(
            samples,
            max_k,
            random_state,
            algorithm=algorithm,
            density_maxk=density_maxk,
            density_z=density_z,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            hierarchical_n_clusters=hierarchical_n_clusters,
            hierarchical_linkage=hierarchical_linkage,
            tomato_k=tomato_k,
            tomato_tau=tomato_tau,
        )
        diag["subsampled"] = False
        diag["subsample_size"] = int(n_frames)
        diag["total_frames"] = int(n_frames)
        diag["assign_k"] = int(assign_k)
        return labels, k, diag, int(n_frames)

    subsample_indices = (
        _uniform_subsample_indices(n_frames, int(max_cluster_frames))
        if subsample_indices is None
        else subsample_indices
    )
    subsample_indices = np.asarray(subsample_indices, dtype=int)
    sub_samples = samples[subsample_indices]
    labels_sub, k, diag = _cluster_residue_samples(
        sub_samples,
        max_k,
        random_state,
        algorithm=algorithm,
        density_maxk=density_maxk,
        density_z=density_z,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        hierarchical_n_clusters=hierarchical_n_clusters,
        hierarchical_linkage=hierarchical_linkage,
        tomato_k=tomato_k,
        tomato_tau=tomato_tau,
    )

    emb_full = _angles_to_embedding(samples)
    emb_sub = emb_full[subsample_indices]
    assigned = _assign_labels_by_knn(emb_full, emb_sub, labels_sub, subsample_indices, assign_k)

    diag["subsampled"] = True
    diag["subsample_size"] = int(subsample_indices.size)
    diag["total_frames"] = int(n_frames)
    diag["assign_k"] = int(min(assign_k, emb_sub.shape[0]))
    return assigned.astype(np.int32), k, diag, int(subsample_indices.size)


def _resolve_states_for_meta(meta: Dict[str, Any], system: SystemMetadata) -> List[DescriptorState]:
    """Return all states contributing to a metastable macro-state."""
    macro_state_id = meta.get("macro_state_id")
    macro_state_name = meta.get("macro_state")
    states: List[DescriptorState] = []
    for st in system.states.values():
        if macro_state_id and st.state_id == macro_state_id:
            states.append(st)
        elif macro_state_name and st.name == macro_state_name:
            states.append(st)
    return states


def _extract_labels_for_state(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    state: DescriptorState,
    features: Dict[str, Any],
) -> np.ndarray:
    """Extract metastable labels array, preferring embedded NPZ key."""
    labels = features.pop("metastable_labels", None)
    if labels is not None:
        labels = np.asarray(labels)
    elif state.metastable_labels_file:
        label_path = store.resolve_path(project_id, system_id, state.metastable_labels_file)
        if label_path.exists():
            labels = np.load(label_path)
    if labels is None:
        raise ValueError(f"No metastable labels found for state '{state.state_id}'.")
    return np.asarray(labels).astype(np.int32)


def _infer_frame_count(features: Dict[str, Any]) -> int:
    """Best-effort frame count inference from descriptor arrays."""
    for key, arr in features.items():
        if key == "metastable_labels":
            continue
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 1:
            return int(arr.shape[0])
    return 0


def _coerce_residue_keys(
    residue_keys: List[str], features: Dict[str, Any], state: DescriptorState
) -> List[str]:
    """Prefer stored residue_keys but fall back to feature keys."""
    if residue_keys:
        return residue_keys
    if state.residue_keys:
        return _sort_residue_keys(state.residue_keys)
    feature_keys = [k for k in features.keys() if k != "metastable_labels"]
    return _sort_residue_keys(feature_keys)


def _sort_residue_keys(keys: List[str]) -> List[str]:
    """Sort keys by numeric resid extracted from patterns like 'res_123'."""
    def _extract_num(k: str) -> int:
        m = re.search(r"(\d+)$", k)
        return int(m.group(1)) if m else 0
    return sorted(keys, key=_extract_num)


def _extract_residue_positions(
    pdb_path: Path,
    residue_keys: List[str],
    residue_mapping: Dict[str, str],
    contact_mode: str,
) -> List[Optional[np.ndarray]]:
    """Return per-residue positions (CA or center-of-mass) for contact computation."""
    positions: List[Optional[np.ndarray]] = []
    u = mda.Universe(str(pdb_path))
    for key in residue_keys:
        sel = residue_mapping.get(key) or key
        try:
            res_atoms = u.select_atoms(sel)
        except Exception:
            positions.append(None)
            continue
        if res_atoms.n_atoms == 0:
            positions.append(None)
            continue
        if contact_mode == "CA":
            ca_atoms = res_atoms.select_atoms("name CA")
            if ca_atoms.n_atoms > 0:
                positions.append(np.array(ca_atoms[0].position, dtype=float))
            else:
                positions.append(np.array(res_atoms.center_of_mass(), dtype=float))
        else:
            positions.append(np.array(res_atoms.center_of_mass(), dtype=float))
    return positions


def _compute_contact_edges(
    pdb_path: Path,
    residue_keys: List[str],
    residue_mapping: Dict[str, str],
    cutoff: float,
    contact_mode: str,
) -> set:
    """Compute contact edges (i,j) for one PDB."""
    positions = _extract_residue_positions(pdb_path, residue_keys, residue_mapping, contact_mode)
    valid_indices = [i for i, pos in enumerate(positions) if pos is not None]
    edges: set = set()
    if len(valid_indices) < 2:
        return edges
    coords = np.stack([positions[i] for i in valid_indices], axis=0)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    for a_idx, i in enumerate(valid_indices):
        for b_idx in range(a_idx + 1, len(valid_indices)):
            j = valid_indices[b_idx]
            if dist[a_idx, b_idx] < cutoff:
                edges.add((min(i, j), max(i, j)))
    return edges


def _collect_cluster_inputs(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
    cutoff_val: float,
    contact_atom_mode: str,
) -> Dict[str, Any]:
    unique_meta_ids = list(dict.fromkeys([str(mid) for mid in metastable_ids]))

    store = ProjectStore()
    system = store.get_system(project_id, system_id)
    macro_only = getattr(system, "analysis_mode", None) == "macro"
    metastable_lookup = {
        m.get("metastable_id"): {**m, "meta_kind": "metastable"} for m in system.metastable_states or []
    }
    if macro_only:
        for st in system.states.values():
            metastable_lookup.setdefault(
                st.state_id,
                {
                    "metastable_id": st.state_id,
                    "metastable_index": 0,
                    "macro_state_id": st.state_id,
                    "macro_state": st.name,
                    "name": st.name,
                    "default_name": st.name,
                    "representative_pdb": st.pdb_file,
                    "meta_kind": "macro",
                },
            )

    residue_keys: List[str] = []
    residue_mapping: Dict[str, str] = {}
    merged_angles_per_residue: List[List[np.ndarray]] = []
    merged_frame_state_ids: List[str] = []
    merged_frame_meta_ids: List[str] = []
    merged_frame_indices: List[int] = []
    contact_edges: set = set()
    contact_sources: List[str] = []

    for meta_id in unique_meta_ids:
        meta = metastable_lookup.get(meta_id)
        if not meta:
            raise ValueError(f"Metastable state '{meta_id}' not found on this system.")
        is_macro = meta.get("meta_kind") == "macro"
        meta_index = meta.get("metastable_index")
        if meta_index is None:
            raise ValueError(f"Metastable state '{meta_id}' is missing its index.")

        candidate_states = _resolve_states_for_meta(meta, system)
        if not candidate_states:
            raise ValueError(f"No descriptor-ready states found for metastable '{meta_id}'.")
        matched_frames = 0

        for state in candidate_states:
            if not state.descriptor_file:
                continue
            desc_path = store.resolve_path(project_id, system_id, state.descriptor_file)
            features = load_descriptor_npz(desc_path)
            if is_macro:
                frame_count = _infer_frame_count(features)
                if frame_count <= 0:
                    raise ValueError(f"Could not determine frame count for macro-state '{state.state_id}'.")
                labels = np.zeros(frame_count, dtype=np.int32)
            else:
                labels = _extract_labels_for_state(store, project_id, system_id, state, features)

            residue_keys = _coerce_residue_keys(residue_keys, features, state)
            if not residue_keys:
                raise ValueError("Could not determine residue keys for clustering.")
            if not residue_mapping:
                residue_mapping = dict(state.residue_mapping or system.residue_selections_mapping or {})

            if not merged_angles_per_residue:
                merged_angles_per_residue = [[] for _ in residue_keys]

            if labels.shape[0] == 0:
                continue
            mask = labels == int(meta_index)
            if not np.any(mask):
                continue

            matched_indices = np.where(mask)[0]
            for idx in matched_indices:
                merged_frame_state_ids.append(state.state_id)
                merged_frame_meta_ids.append(meta_id)
                merged_frame_indices.append(int(idx))
                for col, key in enumerate(residue_keys):
                    arr = np.asarray(features.get(key))
                    if arr is None or arr.shape[0] != labels.shape[0]:
                        raise ValueError(
                            f"Descriptor array for '{key}' is missing or misaligned in state '{state.state_id}'."
                        )
                    if arr.ndim >= 3:
                        vec = arr[idx, 0, :3]
                    elif arr.ndim == 2:
                        vec = arr[idx, :3]
                    else:
                        vec = arr[idx : idx + 1]
                    vec = np.asarray(vec, dtype=float).reshape(-1)
                    if vec.size < 3:
                        padded = np.zeros(3, dtype=float)
                        padded[: vec.size] = vec
                        vec = padded
                    else:
                        vec = vec[:3]
                    merged_angles_per_residue[col].append(vec)
                matched_frames += 1

        if matched_frames == 0:
            raise ValueError(f"No frames matched metastable '{meta_id}'.")

        rep_rel = meta.get("representative_pdb")
        if rep_rel:
            rep_path = store.resolve_path(project_id, system_id, rep_rel)
            if rep_path.exists():
                contact_sources.append(str(rep_path))
                try:
                    edges = _compute_contact_edges(
                        rep_path,
                        residue_keys,
                        residue_mapping,
                        cutoff_val,
                        contact_atom_mode,
                    )
                    contact_edges.update(edges)
                except Exception:
                    pass

    return {
        "unique_meta_ids": unique_meta_ids,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "merged_angles_per_residue": merged_angles_per_residue,
        "merged_frame_state_ids": merged_frame_state_ids,
        "merged_frame_meta_ids": merged_frame_meta_ids,
        "merged_frame_indices": merged_frame_indices,
        "contact_edges": contact_edges,
        "contact_sources": contact_sources,
    }


def generate_metastable_cluster_npz(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
    *,
    max_clusters_per_residue: int = 6,
    max_cluster_frames: Optional[int] = None,
    random_state: int = 0,
    contact_cutoff: float = 10.0,
    contact_atom_mode: str = "CA",
    cluster_algorithm: str = "density_peaks",
    density_maxk: Optional[int] = 100,
    density_z: float | str | None = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    hierarchical_n_clusters: Optional[int] = None,
    hierarchical_linkage: str = "ward",
    tomato_k: int = 15,
    tomato_tau: float | str = "auto",
    tomato_k_max: Optional[int] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Build per-residue cluster labels for selected metastable states and save NPZ.

    Returns the path to the NPZ and a metadata dictionary.
    """
    if not metastable_ids:
        raise ValueError("At least one metastable_id is required.")
    if max_clusters_per_residue < 1:
        raise ValueError("max_clusters_per_residue must be >= 1.")
    max_cluster_frames_val: Optional[int] = None
    if max_cluster_frames is not None:
        try:
            max_cluster_frames_val = int(max_cluster_frames)
        except Exception as exc:
            raise ValueError("max_cluster_frames must be an integer.") from exc
        if max_cluster_frames_val < 1:
            raise ValueError("max_cluster_frames must be >= 1.")
    try:
        cutoff_val = float(contact_cutoff)
    except Exception as exc:
        raise ValueError("contact_cutoff must be a number.") from exc
    if cutoff_val <= 0:
        raise ValueError("contact_cutoff must be > 0.")
    contact_atom_mode = str(contact_atom_mode or "CA").upper()
    if contact_atom_mode not in {"CA", "CM"}:
        raise ValueError("contact_atom_mode must be 'CA' or 'CM'.")
    algo = (cluster_algorithm or "dbscan").lower()
    if algo not in {"tomato", "dbscan", "kmeans", "hierarchical", "density_peaks"}:
        raise ValueError("cluster_algorithm must be one of: tomato, density_peaks, dbscan, kmeans, hierarchical.")
    try:
        db_eps_val = float(dbscan_eps)
    except Exception as exc:
        raise ValueError("dbscan_eps must be a number.") from exc
    try:
        db_min_samples_val = max(1, int(dbscan_min_samples))
    except Exception as exc:
        raise ValueError("dbscan_min_samples must be an integer >=1.") from exc
    if density_maxk is None:
        density_maxk_val = 100
    else:
        try:
            density_maxk_val = max(1, int(density_maxk))
        except Exception as exc:
            raise ValueError("density_maxk must be an integer >=1.") from exc
    if density_z is None or (isinstance(density_z, str) and density_z.lower() == "auto"):
        density_z_val: float | str = "auto"
    else:
        try:
            density_z_val = float(density_z)
        except Exception as exc:
            raise ValueError("density_z must be a number or 'auto'.") from exc
    h_linkage = (hierarchical_linkage or "ward").lower()
    if h_linkage not in {"ward", "complete", "average", "single"}:
        h_linkage = "ward"
    h_n_clusters_val = (
        max_clusters_per_residue if hierarchical_n_clusters is None else max(1, int(hierarchical_n_clusters))
    )
    t_k = max(1, int(tomato_k))
    t_k_max = max_clusters_per_residue if tomato_k_max is None else max(1, int(tomato_k_max))
    if tomato_tau is None or (isinstance(tomato_tau, str) and tomato_tau.lower() == "auto"):
        t_tau = "auto"
    else:
        try:
            t_tau = float(tomato_tau)
        except Exception as exc:
            raise ValueError("tomato_tau must be a number or 'auto'.") from exc

    inputs = _collect_cluster_inputs(project_id, system_id, metastable_ids, cutoff_val, contact_atom_mode)
    unique_meta_ids = inputs["unique_meta_ids"]
    residue_keys = inputs["residue_keys"]
    residue_mapping = inputs["residue_mapping"]
    merged_angles_per_residue = inputs["merged_angles_per_residue"]
    merged_frame_state_ids = inputs["merged_frame_state_ids"]
    merged_frame_meta_ids = inputs["merged_frame_meta_ids"]
    merged_frame_indices = inputs["merged_frame_indices"]
    contact_edges = inputs["contact_edges"]
    contact_sources = inputs["contact_sources"]
    tomato_diag_merged: Optional[Dict[str, Any]] = None
    total_residue_jobs = len(residue_keys)
    completed_residue_jobs = 0
    assign_k = 10
    if progress_callback:
        progress_callback("Clustering residues...", 0, total_residue_jobs)
    store = ProjectStore()

    # Build merged clusters
    if not merged_angles_per_residue:
        raise ValueError("No frames gathered across the selected metastable states.")

    merged_frame_count = len(merged_frame_state_ids)
    merged_labels = np.zeros((merged_frame_count, len(residue_keys)), dtype=np.int32)
    merged_counts = np.zeros(len(residue_keys), dtype=np.int32)
    merged_subsample_indices = None
    if max_cluster_frames_val and merged_frame_count > max_cluster_frames_val:
        merged_subsample_indices = _uniform_subsample_indices(merged_frame_count, max_cluster_frames_val)
    merged_clustered_frames = (
        merged_subsample_indices.size if merged_subsample_indices is not None else merged_frame_count
    )
    for col, samples in enumerate(merged_angles_per_residue):
        sample_arr = np.asarray(samples, dtype=float)
        if sample_arr.shape[0] != merged_frame_count:
            raise ValueError("Merged residue samples have inconsistent frame counts.")
        labels_arr, k, diag, _ = _cluster_with_subsample(
            sample_arr,
            max_clusters_per_residue,
            random_state,
            algorithm=algo,
            density_maxk=density_maxk_val,
            density_z=density_z_val,
            dbscan_eps=db_eps_val,
            dbscan_min_samples=db_min_samples_val,
            hierarchical_n_clusters=h_n_clusters_val,
            hierarchical_linkage=h_linkage,
            tomato_k=t_k,
            tomato_tau=t_tau,
            max_cluster_frames=max_cluster_frames_val,
            assign_k=assign_k,
            subsample_indices=merged_subsample_indices,
        )
        if labels_arr.size == 0:
            merged_labels[:, col] = -1
            merged_counts[col] = 0
        else:
            if np.any(labels_arr >= 0):
                k = max(k, int(labels_arr.max()) + 1)
            merged_labels[:, col] = labels_arr
            merged_counts[col] = k
        if algo == "tomato" and col == 0 and diag and tomato_diag_merged is None:
            tomato_diag_merged = diag
        if progress_callback and total_residue_jobs:
            completed_residue_jobs += 1
            progress_callback(
                f"Clustering residues: {completed_residue_jobs}/{total_residue_jobs}",
                completed_residue_jobs,
                total_residue_jobs,
            )

    # Persist NPZ
    dirs = store.ensure_directories(project_id, system_id)
    cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = "-".join(_slug(mid)[:24] for mid in unique_meta_ids) or "metastable"
    out_path = cluster_dir / f"{suffix}_clusters_{timestamp}.npz"

    metadata = {
        "project_id": project_id,
        "system_id": system_id,
        "generated_at": datetime.utcnow().isoformat(),
        "selected_metastable_ids": unique_meta_ids,
        "cluster_algorithm": algo,
        "cluster_params": {
            "dbscan_eps": db_eps_val,
            "dbscan_min_samples": db_min_samples_val,
            "hierarchical_n_clusters": h_n_clusters_val,
            "hierarchical_linkage": h_linkage,
            "tomato_k": t_k,
            "tomato_tau": t_tau,
            "density_maxk": density_maxk_val,
            "density_z": density_z_val,
            "max_clusters_per_residue": max_clusters_per_residue,
            "max_cluster_frames": max_cluster_frames_val,
            "assign_k": assign_k,
            "random_state": random_state,
        },
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "max_clusters_per_residue": max_clusters_per_residue,
        "random_state": random_state,
        "contact_mode": contact_atom_mode,
        "contact_cutoff": cutoff_val,
        "contact_sources": contact_sources,
        "contact_edge_count": len(contact_edges),
        "merged": {
            "n_frames": merged_frame_count,
            "clustered_frames": int(merged_clustered_frames),
            "npz_keys": {
                "labels": "merged__labels",
                "cluster_counts": "merged__cluster_counts",
                "frame_state_ids": "merged__frame_state_ids",
                "frame_metastable_ids": "merged__frame_metastable_ids",
                "frame_indices": "merged__frame_indices",
            },
        },
    }
    if algo == "tomato":
        tomato_meta = {}
        if tomato_diag_merged:
            tomato_meta["merged_example"] = tomato_diag_merged
        metadata["cluster_params"]["tomato_diagnostics"] = tomato_meta

    payload: Dict[str, Any] = {
        "residue_keys": np.array(residue_keys),
        "metadata_json": np.array(json.dumps(metadata)),
        "merged__labels": merged_labels,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": np.array(merged_frame_state_ids),
        "merged__frame_metastable_ids": np.array(merged_frame_meta_ids),
        "merged__frame_indices": np.array(merged_frame_indices, dtype=np.int64),
    }
    if contact_edges:
        edge_index = np.array(sorted(contact_edges), dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    payload["contact_edge_index"] = edge_index
    payload["contact_mode"] = np.array(contact_atom_mode)
    payload["contact_cutoff"] = np.array(cutoff_val)

    np.savez_compressed(out_path, **payload)
    return out_path, metadata


def prepare_cluster_workspace(
    project_id: str,
    system_id: str,
    metastable_ids: List[str],
    *,
    max_clusters_per_residue: int,
    max_cluster_frames: Optional[int],
    random_state: int,
    contact_cutoff: float,
    contact_atom_mode: str,
    cluster_algorithm: str,
    density_maxk: int,
    density_z: float | str | None,
    dbscan_eps: float,
    dbscan_min_samples: int,
    hierarchical_n_clusters: Optional[int],
    hierarchical_linkage: str,
    tomato_k: int,
    tomato_tau: float | str,
    tomato_k_max: Optional[int],
    work_dir: Path,
) -> Dict[str, Any]:
    """Precompute and persist clustering inputs for fan-out chunk jobs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cutoff_val = float(contact_cutoff)
    contact_atom_mode = str(contact_atom_mode or "CA").upper()

    inputs = _collect_cluster_inputs(project_id, system_id, metastable_ids, cutoff_val, contact_atom_mode)
    residue_keys = inputs["residue_keys"]
    merged_angles_per_residue = inputs["merged_angles_per_residue"]
    merged_frame_state_ids = inputs["merged_frame_state_ids"]
    merged_frame_meta_ids = inputs["merged_frame_meta_ids"]
    merged_frame_indices = inputs["merged_frame_indices"]
    contact_edges = inputs["contact_edges"]
    contact_sources = inputs["contact_sources"]
    residue_mapping = inputs["residue_mapping"]
    unique_meta_ids = inputs["unique_meta_ids"]

    if not merged_angles_per_residue:
        raise ValueError("No frames gathered across the selected metastable states.")

    n_frames = len(merged_frame_state_ids)
    n_residues = len(residue_keys)
    angles_arr = np.stack(
        [np.asarray(samples, dtype=np.float32) for samples in merged_angles_per_residue], axis=1
    )
    angles_path = work_dir / "angles.npy"
    np.save(angles_path, angles_arr)

    np.save(work_dir / "frame_state_ids.npy", np.array(merged_frame_state_ids))
    np.save(work_dir / "frame_meta_ids.npy", np.array(merged_frame_meta_ids))
    np.save(work_dir / "frame_indices.npy", np.array(merged_frame_indices, dtype=np.int64))

    if contact_edges:
        edge_index = np.array(sorted(contact_edges), dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    np.save(work_dir / "contact_edge_index.npy", edge_index)
    np.save(work_dir / "contact_mode.npy", np.array(contact_atom_mode))
    np.save(work_dir / "contact_cutoff.npy", np.array(cutoff_val))

    manifest = {
        "project_id": project_id,
        "system_id": system_id,
        "selected_metastable_ids": unique_meta_ids,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "n_frames": n_frames,
        "n_residues": n_residues,
        "angles_path": "angles.npy",
        "frame_state_ids_path": "frame_state_ids.npy",
        "frame_meta_ids_path": "frame_meta_ids.npy",
        "frame_indices_path": "frame_indices.npy",
        "contact_edge_index_path": "contact_edge_index.npy",
        "contact_mode_path": "contact_mode.npy",
        "contact_cutoff_path": "contact_cutoff.npy",
        "contact_sources": contact_sources,
        "cluster_algorithm": cluster_algorithm,
        "cluster_params": {
            "dbscan_eps": float(dbscan_eps),
            "dbscan_min_samples": int(dbscan_min_samples),
            "hierarchical_n_clusters": int(hierarchical_n_clusters)
            if hierarchical_n_clusters is not None
            else None,
            "hierarchical_linkage": str(hierarchical_linkage or "ward"),
            "tomato_k": int(tomato_k),
            "tomato_tau": tomato_tau,
            "density_maxk": int(density_maxk),
            "density_z": density_z,
            "max_clusters_per_residue": int(max_clusters_per_residue),
            "max_cluster_frames": int(max_cluster_frames) if max_cluster_frames else None,
            "assign_k": 10,
            "random_state": int(random_state),
            "tomato_k_max": int(tomato_k_max) if tomato_k_max is not None else None,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run_cluster_chunk(
    work_dir: Path,
    residue_index: int,
) -> Dict[str, Any]:
    """Run clustering for a single residue and persist labels."""
    manifest_path = work_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    angles_path = work_dir / manifest["angles_path"]
    angles = np.load(angles_path, mmap_mode="r")
    n_frames, n_residues, _ = angles.shape
    if residue_index < 0 or residue_index >= n_residues:
        raise ValueError(f"Residue index {residue_index} out of range (0..{n_residues - 1}).")

    params = manifest.get("cluster_params", {})
    algo = manifest.get("cluster_algorithm", "density_peaks")
    sample_arr = np.asarray(angles[:, residue_index, :], dtype=float)
    labels, k, diag, _ = _cluster_with_subsample(
        sample_arr,
        int(params.get("max_clusters_per_residue", 6)),
        int(params.get("random_state", 0)),
        algorithm=algo,
        density_maxk=int(params.get("density_maxk", 100)),
        density_z=params.get("density_z", "auto"),
        dbscan_eps=float(params.get("dbscan_eps", 0.5)),
        dbscan_min_samples=int(params.get("dbscan_min_samples", 5)),
        hierarchical_n_clusters=params.get("hierarchical_n_clusters"),
        hierarchical_linkage=params.get("hierarchical_linkage", "ward"),
        tomato_k=int(params.get("tomato_k", 15)),
        tomato_tau=params.get("tomato_tau", "auto"),
        max_cluster_frames=params.get("max_cluster_frames"),
        assign_k=int(params.get("assign_k", 10)),
    )

    out_path = work_dir / f"chunk_{residue_index:04d}.npz"
    payload: Dict[str, Any] = {
        "labels": labels.astype(np.int32),
        "cluster_count": np.array([int(k)], dtype=np.int32),
    }
    if residue_index == 0 and algo == "tomato":
        payload["diagnostics_json"] = np.array(json.dumps(diag))

    np.savez_compressed(out_path, **payload)
    return {"residue_index": residue_index, "path": str(out_path)}


def reduce_cluster_workspace(work_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    """Combine chunk outputs into a final cluster NPZ."""
    manifest = json.loads((work_dir / "manifest.json").read_text())
    project_id = manifest["project_id"]
    system_id = manifest["system_id"]
    residue_keys = manifest["residue_keys"]
    residue_mapping = manifest.get("residue_mapping") or {}
    n_frames = int(manifest.get("n_frames", 0))
    n_residues = int(manifest.get("n_residues", len(residue_keys)))
    selected_meta_ids = manifest.get("selected_metastable_ids") or []
    cluster_params = manifest.get("cluster_params") or {}
    algo = manifest.get("cluster_algorithm") or "density_peaks"

    merged_labels = np.zeros((n_frames, n_residues), dtype=np.int32)
    merged_counts = np.zeros(n_residues, dtype=np.int32)
    tomato_diag_merged = None

    for idx in range(n_residues):
        chunk_path = work_dir / f"chunk_{idx:04d}.npz"
        if not chunk_path.exists():
            raise ValueError(f"Missing cluster chunk output for residue index {idx}.")
        data = np.load(chunk_path, allow_pickle=True)
        labels = np.asarray(data["labels"], dtype=np.int32)
        if labels.shape[0] != n_frames:
            raise ValueError(f"Chunk {idx} has {labels.shape[0]} frames, expected {n_frames}.")
        merged_labels[:, idx] = labels
        merged_counts[idx] = int(np.asarray(data["cluster_count"]).reshape(-1)[0])
        if idx == 0 and "diagnostics_json" in data:
            try:
                tomato_diag_merged = json.loads(str(data["diagnostics_json"].item()))
            except Exception:
                tomato_diag_merged = None

    max_cluster_frames_val = cluster_params.get("max_cluster_frames")
    if max_cluster_frames_val and n_frames > int(max_cluster_frames_val):
        merged_clustered_frames = len(_uniform_subsample_indices(n_frames, int(max_cluster_frames_val)))
    else:
        merged_clustered_frames = n_frames

    store = ProjectStore()
    dirs = store.ensure_directories(project_id, system_id)
    cluster_dir = dirs["system_dir"] / "metastable" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    suffix = "-".join(_slug(mid)[:24] for mid in selected_meta_ids) or "metastable"
    out_path = cluster_dir / f"{suffix}_clusters_{timestamp}.npz"

    frame_state_ids = np.load(work_dir / manifest["frame_state_ids_path"], allow_pickle=True)
    frame_meta_ids = np.load(work_dir / manifest["frame_meta_ids_path"], allow_pickle=True)
    frame_indices = np.load(work_dir / manifest["frame_indices_path"], allow_pickle=True)
    contact_edge_index = np.load(work_dir / manifest["contact_edge_index_path"], allow_pickle=True)
    contact_mode = np.load(work_dir / manifest["contact_mode_path"], allow_pickle=True)
    contact_cutoff = np.load(work_dir / manifest["contact_cutoff_path"], allow_pickle=True)
    contact_sources = manifest.get("contact_sources") or []

    metadata = {
        "project_id": project_id,
        "system_id": system_id,
        "generated_at": datetime.utcnow().isoformat(),
        "selected_metastable_ids": selected_meta_ids,
        "cluster_algorithm": algo,
        "cluster_params": cluster_params,
        "residue_keys": residue_keys,
        "residue_mapping": residue_mapping,
        "max_clusters_per_residue": cluster_params.get("max_clusters_per_residue"),
        "random_state": cluster_params.get("random_state"),
        "contact_mode": str(contact_mode.item() if hasattr(contact_mode, "item") else contact_mode),
        "contact_cutoff": float(contact_cutoff.item() if hasattr(contact_cutoff, "item") else contact_cutoff),
        "contact_sources": contact_sources,
        "contact_edge_count": int(contact_edge_index.shape[1]) if contact_edge_index.ndim == 2 else 0,
        "merged": {
            "n_frames": n_frames,
            "clustered_frames": int(merged_clustered_frames),
            "npz_keys": {
                "labels": "merged__labels",
                "cluster_counts": "merged__cluster_counts",
                "frame_state_ids": "merged__frame_state_ids",
                "frame_metastable_ids": "merged__frame_metastable_ids",
                "frame_indices": "merged__frame_indices",
            },
        },
    }
    if algo == "tomato" and tomato_diag_merged:
        metadata["cluster_params"]["tomato_diagnostics"] = {"merged_example": tomato_diag_merged}

    payload = {
        "residue_keys": np.array(residue_keys),
        "metadata_json": np.array(json.dumps(metadata)),
        "merged__labels": merged_labels,
        "merged__cluster_counts": merged_counts,
        "merged__frame_state_ids": frame_state_ids,
        "merged__frame_metastable_ids": frame_meta_ids,
        "merged__frame_indices": frame_indices,
        "contact_edge_index": contact_edge_index,
        "contact_mode": np.array(contact_mode),
        "contact_cutoff": np.array(contact_cutoff),
    }

    np.savez_compressed(out_path, **payload)
    return out_path, metadata
