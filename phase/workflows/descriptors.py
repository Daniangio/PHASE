"""Descriptor pre-processing pipeline for uploaded trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from phase.features.extraction import FeatureDict, FeatureExtractor
from phase.io.readers import MDAnalysisReader
from phase.pipeline.builder import DatasetBuilder
from phase.pipeline.selection_parser import parse_and_expand_selections


@dataclass
class DescriptorBuildResult:
    active_features: FeatureDict
    inactive_features: FeatureDict
    n_frames_active: int
    n_frames_inactive: int
    residue_keys: List[str]
    residue_mapping: Dict[str, str]


@dataclass
class SingleDescriptorBuildResult:
    features: FeatureDict
    n_frames: int
    residue_keys: List[str]
    residue_mapping: Dict[str, str]


SelectionInput = Union[Dict[str, str], List[str]]


class DescriptorPreprocessor:
    """
    Wraps the DatasetBuilder to extract the phi/psi/chi1 descriptors once,
    without persisting the original trajectory files.
    """

    def __init__(self, residue_selections: Optional[SelectionInput] = None):
        reader = MDAnalysisReader()
        extractor = FeatureExtractor(residue_selections)
        self.reader = reader
        self.extractor = extractor
        self.builder = DatasetBuilder(reader, extractor)

    @staticmethod
    def _filter_degenerate_features(features: FeatureDict) -> Tuple[FeatureDict, List[str]]:
        """
        Drop residue features that are empty, non-finite, or all-zero to avoid degenerate descriptors.
        Returns the cleaned features and a list of dropped residue keys.
        """
        cleaned: FeatureDict = {}
        dropped: List[str] = []
        for key, arr in features.items():
            arr_np = np.asarray(arr)
            if arr_np.size == 0 or not np.all(np.isfinite(arr_np)) or np.allclose(arr_np, 0.0):
                dropped.append(key)
                continue
            cleaned[key] = arr
        return cleaned, dropped

    def build(
        self,
        active_traj: str,
        active_topo: str,
        inactive_traj: str,
        inactive_topo: str,
        active_slice: Optional[str] = None,
        inactive_slice: Optional[str] = None,
    ) -> DescriptorBuildResult:
        (
            features_active,
            n_frames_active,
            features_inactive,
            n_frames_inactive,
            common_keys,
            mapping,
        ) = self.builder._parallel_load_and_extract(  # pylint: disable=protected-access
            active_traj,
            active_topo,
            inactive_traj,
            inactive_topo,
            active_slice=active_slice,
            inactive_slice=inactive_slice,
        )

        # Remove degenerate residues before proceeding
        features_active, dropped_active = self._filter_degenerate_features(features_active)
        features_inactive, dropped_inactive = self._filter_degenerate_features(features_inactive)

        filtered_common = (
            set(features_active.keys())
            & set(features_inactive.keys())
            & set(common_keys)
        )
        if not filtered_common:
            raise ValueError(
                f"No valid residues after filtering degenerate features "
                f"(dropped active: {len(dropped_active)}, dropped inactive: {len(dropped_inactive)})."
            )

        active_common = {k: features_active[k] for k in filtered_common}
        inactive_common = {k: features_inactive[k] for k in filtered_common}

        return DescriptorBuildResult(
            active_features=active_common,
            inactive_features=inactive_common,
            n_frames_active=n_frames_active,
            n_frames_inactive=n_frames_inactive,
            residue_keys=sorted(filtered_common),
            residue_mapping={k: v for k, v in mapping.items() if k in filtered_common},
        )

    def build_single(
        self,
        traj_path: str,
        topo_path: str,
        slice_spec: Optional[str] = None,
    ) -> SingleDescriptorBuildResult:
        """
        Extracts descriptors for a single state.
        """
        universe = self.reader.load_trajectory(traj_path, topo_path)
        if universe is None:
            raise ValueError("Failed to load trajectory for descriptor extraction.")

        selections = parse_and_expand_selections(universe, self.extractor.residue_selections)
        if not selections:
            raise ValueError("No residues matched the configured selections for this state.")

        extractor = FeatureExtractor(selections)
        slice_obj = self.builder._parse_slice(slice_spec)  # pylint: disable=protected-access
        features, n_frames = extractor.extract_all_features(universe, slice_obj)
        if not features or n_frames == 0:
            raise ValueError("Descriptor extraction returned no frames.")

        features, dropped = self._filter_degenerate_features(features)
        if not features:
            raise ValueError("All residue descriptors are degenerate (nan/inf/zero).")

        residue_keys = sorted(features.keys())
        residue_mapping = {key: selections[key] for key in residue_keys}

        return SingleDescriptorBuildResult(
            features=features,
            n_frames=n_frames,
            residue_keys=residue_keys,
            residue_mapping=residue_mapping,
        )
