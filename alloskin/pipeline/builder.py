"""
Implements the Strategy Pattern for data preparation.

Refactored to include:
- **Alignment-first approach**: Loads both universes, performs a sequence
  alignment, and generates a filtered, common set of residues *before*
  any feature extraction.
- This fixes a critical bug where non-equivalent residues could be compared.
- Parallelism is maintained by loading in parallel, then extracting in parallel.
"""

import collections
import numpy as np
from typing import Tuple, Dict, Optional, Set
import concurrent.futures
import Bio.Align

# We need the other components for the new logic
# Assuming they are in the same module directory
from alloskin.io.readers import AbstractTrajectoryReader
from alloskin.features.extraction import FeatureExtractor, FeatureDict

def sequence_alignment(mobile, reference):
    """
    Aligns two MDAnalysis ResidueGroups using Bio.Align.
    
    Args:
        reference: The reference ResidueGroup (e.g., inactive state).
        mobile: The mobile ResidueGroup (e.g., active state).
        
    Returns:
        A tuple of (aligned_mobile_seq, aligned_reference_seq)
    """
    residue_map = {
        "ACE": "E", "NME": "M",
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "CYX": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        # Add other non-canonical amino acids and ligands if needed
        "HOH": "W", # Example for water
    }
    
    def get_seq(resgroup):
        """Helper to convert resnames to 1-letter code, skipping unknowns."""
        seq = []
        for r in resgroup.resnames:
            if r in residue_map:
                seq.append(residue_map[r])
            else:
                print(f"  Warning: Skipping unknown residue '{r}' in alignment.")
        return ''.join(seq)

    aligner = Bio.Align.PairwiseAligner(
        mode="global",
        match_score=2,
        mismatch_score=-1,
        open_gap_score=-2,
        extend_gap_score=-0.1)
        
    # Get 1-letter code sequences
    # Note: reference.resnames and mobile.resnames might have different lengths
    # from the *actual* protein if non-standard residues were skipped.
    # We must use the *original* residue groups (inact_prot_res, act_prot_res)
    # to build the final map, as they have the correct indices.
    ref_seq_str = get_seq(reference)
    mob_seq_str = get_seq(mobile)
    
    if not ref_seq_str or not mob_seq_str:
        print("  Error: Could not generate sequence strings for alignment.")
        return None, None

    try:
        aln = aligner.align(mob_seq_str, ref_seq_str)
        # choose top alignment with highest score
        topalignment = aln[0]
        return topalignment.indices # Returns tuple with np.array of indices of seqA and np.array of indices of seqB
    except Exception as e:
        print(f"  Error during Bio.Align: {e}")
        return None, None


class DatasetBuilder:
    """
    Handles the CRITICAL step of preparing data for different goals.
    Uses a ThreadPoolExecutor to parallelize I/O and CPU tasks.
    """
    def __init__(self, reader: AbstractTrajectoryReader, extractor: FeatureExtractor):
        self.reader = reader
        # The 'extractor' is now a prototype, holding the *original* config selections
        self.config_extractor = extractor

    def _parse_slice(self, slice_str: Optional[str]) -> slice:
        """Converts a string 'start:stop:step' into a slice object."""
        # ... existing code ...
        if not slice_str:
            return slice(None) # Returns slice(None, None, None)
        try:
            parts = [int(p) if p else None for p in slice_str.split(':')]
            if len(parts) > 3:
                raise ValueError("Slice string can have at most 3 parts (start:stop:step).")
            parts.extend([None] * (3 - len(parts)))
            return slice(parts[0], parts[1], parts[2])
        except ValueError as e:
            print(f"  Error: Invalid slice string '{slice_str}'. Must be 'start:stop:step'. Using full trajectory. Error: {e}")
            return slice(None)

    def _get_aligned_selections(
        self, u_act, u_inact
    ):
        """
        Performs sequence alignment and filters the config selections
        to only include residues present and aligned in both states.
        
        Returns:
            (active_selections, inactive_selections):
            Two new selection dictionaries for the FeatureExtractors.
        """
        print("--- Performing Sequence Alignment (Active to Inactive) ---")
        
        try:
            # Select all protein residues for sequence alignment
            inact_prot_res = u_inact.select_atoms('protein').residues
            act_prot_res = u_act.select_atoms('protein').residues
        except Exception as e:
            print(f"  Error selecting protein residues for alignment: {e}")
            return {}, {}

        if len(inact_prot_res) == 0 or len(act_prot_res) == 0:
            print("  Error: No protein residues found in one or both universes. Cannot align.")
            return {}, {}

        # Align active (mobile) *to* inactive (reference)
        # This returns (aligned_active_string, aligned_inactive_string)
        indices_act, indices_inact = sequence_alignment(act_prot_res, inact_prot_res)

        if indices_act is None or indices_inact is None:
            print("  Error: Alignment failed. Cannot filter selections.")
            return {}, {}
        
        # Build a map from active residue index to inactive residue index for aligned pairs
        alignment_map = {}
        for i, act_idx in enumerate(indices_act):
            inact_idx = indices_inact[i]
            # A value of -1 from Bio.Align indicates a gap
            if act_idx != -1 and inact_idx != -1:
                # Check if residues at these indices are the same type
                if act_prot_res[act_idx].resname == inact_prot_res[inact_idx].resname:
                    alignment_map[act_idx] = inact_idx

        print(f"  Alignment produced {len(alignment_map)} matching residue pairs.")

        # Now, filter the *original* config selections based on this alignment map
        active_selections_final = {}
        inactive_selections_final = {}

        # We iterate through the original selections requested by the user
        for key, selection_str in self.config_extractor.residue_selections.items():
            try:
                # Select the residue(s) in the active trajectory based on the user's string
                selected_ag = u_act.select_atoms(selection_str)
                if selected_ag.n_residues == 1:
                    res_act = selected_ag.residues[0]
                    # Is this residue part of our valid alignment?
                    if res_act.ix in alignment_map:
                        inact_res_ix = alignment_map[res_act.ix]
                        res_inact = inact_prot_res[inact_res_ix]
                        
                        # Create new, specific selection strings using resid
                        active_selections_final[key] = f"resid {res_act.resid}"
                        inactive_selections_final[key] = f"resid {res_inact.resid}"
            except Exception as e:
                print(f"  Warning: Could not process selection '{selection_str}' for key '{key}'. Skipping. Error: {e}")
        

        print(f"--- Alignment & Filtering Complete: {len(active_selections_final)} common residues selected ---")
        return active_selections_final, inactive_selections_final


    def _parallel_load_and_extract(
        self,
        active_traj_file: str, active_topo_file: str,
        inactive_traj_file: str, inactive_topo_file: str,
        active_slice: Optional[str] = None,
        inactive_slice: Optional[str] = None
    ) -> Tuple[FeatureDict, int, FeatureDict, int, Set[str]]:
        """
        Internal method to manage the full parallel pipeline:
        1. Parallel Load
        2. Serial Align & Filter
        3. Parallel Extract
        """
        act_slice_obj = self._parse_slice(active_slice)
        inact_slice_obj = self._parse_slice(inactive_slice)
        
        # --- 1. Parallel Load (I/O Bound) ---
        print("Submitting parallel trajectory load tasks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_active = executor.submit(
                self.reader.load_trajectory, 
                active_traj_file, active_topo_file
            )
            future_inactive = executor.submit(
                self.reader.load_trajectory, 
                inactive_traj_file, inactive_topo_file
            )
            
            print("Waiting for load tasks...")
            u_act = future_active.result()
            u_inact = future_inactive.result()
        
        if u_act is None or u_inact is None:
            raise ValueError("Trajectory loading failed for one or both states.")
        
        print("Trajectory loading complete.")

        # --- 2. Serial Align & Filter (Fast, CPU) ---
        active_selections, inactive_selections = self._get_aligned_selections(u_act, u_inact)
        
        if not active_selections:
            raise ValueError("No common aligned residues found based on config. Cannot proceed.")
            
        # --- 3. Parallel Extract (CPU Bound) ---
        extractor_act = FeatureExtractor(active_selections)
        extractor_inact = FeatureExtractor(inactive_selections)
        
        print("Submitting parallel feature extraction tasks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_act_extract = executor.submit(
                extractor_act.extract_all_features, 
                u_act, act_slice_obj
            )
            future_inact_extract = executor.submit(
                extractor_inact.extract_all_features, 
                u_inact, inact_slice_obj
            )

            print("Waiting for extraction tasks...")
            features_active, n_frames_active = future_act_extract.result()
            features_inactive, n_frames_inactive = future_inact_extract.result()

        print("Feature extraction complete.")
        
        # Final check: only return keys present in *both* extracted sets
        common_keys = set(features_active.keys()) & set(features_inactive.keys())
        
        if not common_keys:
             raise ValueError("Feature extraction succeeded but no common keys were found. Check extractor logic.")
        
        print(f"Found {len(common_keys)} common features after extraction.")

        return features_active, n_frames_active, features_inactive, n_frames_inactive, common_keys


    def prepare_static_analysis_data(
        self,
        active_traj_file: str, active_topo_file: str,
        inactive_traj_file: str, inactive_topo_file: str,
        active_slice: Optional[str] = None,
        inactive_slice: Optional[str] = None
    ) -> Tuple[FeatureDict, np.ndarray]:
        """
        Prepares data for GOAL 1 and GOAL 2 (Static Analysis).
        - Loads/aligns/extracts active and inactive trajectories concurrently.
        - Concatenates *aligned* features into a single (time-scrambled) dataset.
        - Creates a corresponding binary state label vector Y.
        """
        print("\n--- Preparing STATIC Analysis Dataset (Goals 1 & 2) ---")
        
        features_active, n_frames_active, \
        features_inactive, n_frames_inactive, \
        common_keys = self._parallel_load_and_extract(
            active_traj_file, active_topo_file,
            inactive_traj_file, inactive_topo_file,
            active_slice, inactive_slice
        )
            
        if n_frames_active == 0 or n_frames_inactive == 0:
            raise ValueError("Extraction failed for one or both states (0 frames returned).")

        n_total_frames = n_frames_active + n_frames_inactive
        print(f"Tasks complete. Active sliced frames: {n_frames_active}, Inactive sliced frames: {n_frames_inactive}, Total: {n_total_frames}")

        labels_Y = np.concatenate([
            np.ones(n_frames_active, dtype=int),    # 1 = Active
            np.zeros(n_frames_inactive, dtype=int)  # 0 = Inactive
        ])

        all_features_static: FeatureDict = {}
        # Only iterate over keys guaranteed to be in both
        for res_key in common_keys:
            all_features_static[res_key] = np.concatenate(
                [features_active[res_key], features_inactive[res_key]],
                axis=0
            )
            print(f"Concatenated features for {res_key}: {all_features_static[res_key].shape}")

        print("Shuffling concatenated dataset to break time-correlations...")
        shuffle_indices = np.random.permutation(n_total_frames)
        labels_Y = labels_Y[shuffle_indices]
        for res_key in all_features_static:
            all_features_static[res_key] = all_features_static[res_key][shuffle_indices]
        
        print("Static dataset prepared and shuffled successfully.")
        return all_features_static, labels_Y

    def prepare_dynamic_analysis_data(
        self,
        active_traj_file: str, active_topo_file: str,
        inactive_traj_file: str, inactive_topo_file: str,
        active_slice: Optional[str] = None,
        inactive_slice: Optional[str] = None
    ) -> Tuple[FeatureDict, FeatureDict]:
        """
        Prepares data for GOAL 3 (Dynamic Analysis).
        - Loads/aligns/extracts active and inactive trajectories concurrently.
        - Returns two SEPARATE, TIME-ORDERED, and ALIGNED feature sets.
        """
        print("\n--- Preparing DYNAMIC Analysis Dataset (Goal 3) ---")

        features_active, n_frames_active, \
        features_inactive, n_frames_inactive, \
        common_keys = self._parallel_load_and_extract(
            active_traj_file, active_topo_file,
            inactive_traj_file, inactive_topo_file,
            active_slice, inactive_slice
        )

        if n_frames_active == 0 or n_frames_inactive == 0:
            raise ValueError("Extraction failed for one or both states (0 frames returned).")

        # Filter dictionaries to only contain common keys
        final_features_active = {k: features_active[k] for k in common_keys}
        final_features_inactive = {k: features_inactive[k] for k in common_keys}

        print(f"Dynamic datasets prepared. Active frames: {n_frames_active}, Inactive frames: {n_frames_inactive}. Common residues: {len(common_keys)}")
        
        return final_features_active, final_features_inactive