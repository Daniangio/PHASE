"""
Handles calculation of dihedral angles and the mandatory
sin/cos transformation.

Refactored to:
- Accept a slice_obj to perform analysis on trajectory slices.
- Use the slice parameters in the Dihedral.run() method.
- Return the number of frames processed.
"""

import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals as mda_dihedrals
import numpy as np
from typing import Dict, List, Optional, Tuple

# Mock types for demonstration
TrajectoryObject = mda.Universe
FeatureDict = Dict[str, np.ndarray]

def deg2rad(deg_array: np.ndarray) -> np.ndarray:
    """Converts an array of degrees to radians."""
    return np.deg2rad(deg_array)

class FeatureExtractor:
    """
    Handles calculation of backbone (phi, psi) and sidechain (chi1)
    dihedral angles and their sin/cos transformation.
    """
    def __init__(self, residue_selections: Optional[Dict[str, str]] = None):
        """
        Initializes with a dictionary of residue selections.
        Example: {'res_50': 'resid 50', 'res_131': 'resid 131'}
        """
        self.residue_selections = residue_selections if residue_selections is not None else {}
        if residue_selections is not None:
            print(f"FeatureExtractor initialized for {len(self.residue_selections)} residues.")
        else:
            print("FeatureExtractor initialized in automatic mode (will use all residues).")

    def _transform_to_circular(self, angles_rad: np.ndarray) -> np.ndarray:
        """
        Transforms an array of angles (n_frames, n_residues, n_angles)
        into the 2D vector representation [sin(th), cos(th)].
        Output shape: (n_frames, n_residues, n_angles * 2)
        """
        sin_transformed = np.sin(angles_rad)
        cos_transformed = np.cos(angles_rad)
        
        n_frames, n_residues, n_angles = angles_rad.shape
        circular_features = np.empty((n_frames, n_residues, n_angles * 2))
        
        circular_features[..., 0::2] = sin_transformed
        circular_features[..., 1::2] = cos_transformed
        
        return circular_features

    def _get_sliced_length(self, traj: TrajectoryObject, slice_obj: slice) -> int:
        """Helper to calculate frames in a slice."""
        total_frames = len(traj.trajectory)
        # Get slice parameters, defaulting to full trajectory
        start = slice_obj.start or 0
        stop = slice_obj.stop or total_frames
        step = slice_obj.step or 1
        
        # Use range to correctly calculate the number of items
        return len(range(start, stop, step))

    def _calculate_dihedral_angle(
        self, 
        atom_groups_list: List[Optional[mda.AtomGroup]], 
        n_frames: int,
        slice_obj: slice
    ) -> np.ndarray:
        """
        Helper function to run Dihedral analysis on a list of AtomGroups
        over a specified trajectory slice.
        
        Args:
            atom_groups_list: List of AtomGroups (or None) to analyze.
            n_frames: The *sliced* number of frames, for result array shape.
            slice_obj: The slice object with start/stop/step.
            
        Returns: (n_sliced_frames, len(atom_groups_list)) array of angles.
        """
        mask = np.array([ag is not None for ag in atom_groups_list])
        angles = np.full((n_frames, len(atom_groups_list)), 0.0, dtype=np.float32)
        
        valid_atom_groups = [ag for ag in atom_groups_list if ag is not None]
        
        if not valid_atom_groups:
            return angles
            
        try:
            # Unpack slice parameters for the .run() method
            start = slice_obj.start
            stop = slice_obj.stop
            step = slice_obj.step
            
            # Run analysis only on the valid groups and the specified slice
            analysis = mda_dihedrals.Dihedral(valid_atom_groups).run(
                start=start, 
                stop=stop, 
                step=step
            )
            
            # Verify shape
            if analysis.results.angles.shape[0] != n_frames:
                print(f"  Warning: Dihedral analysis returned {analysis.results.angles.shape[0]} frames, expected {n_frames}. Check slice logic.")
                # Truncate or handle as needed; for now, we'll trust the mask
            
            angles[:, mask] = analysis.results.angles
            
        except Exception as e:
            print(f"    Warning: Dihedral calculation failed: {e}")
            
        return angles


    def extract_features_for_residues(
        self,
        traj: TrajectoryObject,
        protein_residues: mda.ResidueGroup,
        slice_obj: slice = slice(None)
    ) -> Tuple[Optional[Dict[int, np.ndarray]], int]:
        """
        Performs the expensive, one-time feature extraction for all residues
        in the provided ResidueGroup. This is the core computational step.

        Args:
            traj: The MDAnalysis Universe object.
            protein_residues: A ResidueGroup (e.g., from u.select_atoms('protein').residues)
                              for which to calculate all features.
            slice_obj: A slice object for the trajectory.

        Returns:
            - A dictionary mapping residue index (.ix) to its feature array (n_frames, 1, 6).
            - The number of frames processed.
        """
        n_frames = self._get_sliced_length(traj, slice_obj)
        if n_frames == 0:
            print("      Warning: Slice results in 0 frames. Skipping extraction.")
            return None, 0

        print(f"    Calculating dihedrals for {len(protein_residues)} residues in one pass...")

        try:
            # Get selections for ALL residues at once
            phi_selections = protein_residues.phi_selections()
            psi_selections = protein_residues.psi_selections()
            chi1_selections = protein_residues.chi1_selections()

            # We need to know which residue corresponds to which angle
            phi_res_indices = [res.ix for res in protein_residues if res.phi_selection() is not None]
            psi_res_indices = [res.ix for res in protein_residues if res.psi_selection() is not None]
            chi1_res_indices = [res.ix for res in protein_residues if res.chi1_selection() is not None]

            all_selections = phi_selections + psi_selections + chi1_selections
            if not all_selections:
                print("      Warning: No valid dihedrals found for the entire protein selection.")
                return None, 0

            # --- Single, expensive calculation over the trajectory slice ---
            all_angles_deg = self._calculate_dihedral_angle(all_selections, n_frames, slice_obj)
            # -------------------------------------------------------------

            # De-multiplex the results
            n_phi = len(phi_selections)
            n_psi = len(psi_selections)
            phi_angles_all = all_angles_deg[:, :n_phi]
            psi_angles_all = all_angles_deg[:, n_phi : n_phi + n_psi]
            chi1_angles_all = all_angles_deg[:, n_phi + n_psi :]

            # Create a per-residue dictionary to store results
            # Shape: (n_frames, 3) for phi, psi, chi1
            res_angle_map = {res.ix: np.zeros((n_frames, 3)) for res in protein_residues}

            # Populate the map
            for i, res_ix in enumerate(phi_res_indices):
                res_angle_map[res_ix][:, 0] = phi_angles_all[:, i]
            for i, res_ix in enumerate(psi_res_indices):
                res_angle_map[res_ix][:, 1] = psi_angles_all[:, i]
            for i, res_ix in enumerate(chi1_res_indices):
                res_angle_map[res_ix][:, 2] = chi1_angles_all[:, i]

            # Now, transform to circular coordinates and store in the final dict
            all_residue_features: Dict[int, np.ndarray] = {}
            for res_ix, angles_deg in res_angle_map.items():
                # Reshape to (n_frames, 1, 3) to match transform function's expectation
                angles_deg_reshaped = angles_deg[:, np.newaxis, :]
                angles_rad = np.deg2rad(angles_deg_reshaped)
                # Resulting shape is (n_frames, 1, 6)
                all_residue_features[res_ix] = self._transform_to_circular(angles_rad)

            print(f"      Bulk extraction complete. Found features for {len(all_residue_features)} residues.")
            return all_residue_features, n_frames

        except Exception as e:
            print(f"    FATAL ERROR during bulk feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return None, 0


    def extract_all_features(
        self, 
        traj: TrajectoryObject, 
        slice_obj: slice = slice(None)
    ) -> Tuple[FeatureDict, int]:
        """
        Extracts features for all defined residues on a given slice.

        **MODIFIED**: This method now acts as a filter. It expects to be called
        AFTER the expensive `extract_features_for_residues` has been run. It
        selects the pre-computed features based on `self.residue_selections`.

        Args:
            traj: The MDAnalysis Universe object.
            slice_obj: A slice object (e.g., slice(1000, 5000, 2)).

        Returns:
            - A dictionary {selection_key: feature_array}
              where each array has shape (n_sliced_frames, n_residues, 6).
            - The number of frames actually processed (n_sliced_frames).
        """
        if not self.residue_selections:
            print("  Warning: FeatureExtractor has no residue selections. Returning empty results.")
            return {}, 0

        # --- This is the new, efficient "bulk" calculation ---
        # OPTIMIZATION: Instead of selecting the whole protein, we build a
        # single selection string for only the residues we need.
        combined_selection_string = "protein and (" + " or ".join(f"({sel})" for sel in self.residue_selections.values()) + ")"
        
        # This creates a ResidueGroup of only the aligned, filtered residues.
        target_residues = traj.select_atoms(combined_selection_string).residues
        
        # Now, we run the expensive calculation ONLY on this small group.
        all_residue_features, n_frames = self.extract_features_for_residues(
            traj, target_residues, slice_obj
        )

        if all_residue_features is None:
            return {}, 0

        # --- This section now just filters the pre-computed results ---
        filtered_features_dict: FeatureDict = {}
        print("  Filtering pre-computed features based on config selections...")
        for key, sel_string in self.residue_selections.items():
            try:
                # Select all residues in the group to get their indices
                target_residues_group = traj.select_atoms(sel_string).residues
                
                if target_residues_group.n_residues > 0:
                    # Collect the pre-computed features for each residue in the group
                    feature_list_for_group = []
                    all_found = True
                    for res in target_residues_group:
                        if res.ix in all_residue_features:
                            feature_list_for_group.append(all_residue_features[res.ix])
                        else:
                            print(f"    Warning: Feature for resid {res.resid} (part of group '{key}') not found. Skipping group.")
                            all_found = False
                            break
                    
                    # If all features were found, concatenate them
                    if all_found and feature_list_for_group:
                        # Concatenate along the last axis (the feature dimension)
                        # (n_frames, 1, 6), (n_frames, 1, 6) -> (n_frames, 1, 12)
                        filtered_features_dict[key] = np.concatenate(feature_list_for_group, axis=-1)
                else:
                    print(f"    Warning: Selection '{sel_string}' for key '{key}' resolved to 0 residues. Skipping.")
                    continue
            except Exception as e:
                print(f"    Warning: Could not resolve selection '{sel_string}' for key '{key}'. Skipping. Error: {e}")

        return filtered_features_dict, n_frames