"""Descriptor pre-processing pipeline for uploaded trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from alloskin.features.extraction import FeatureDict, FeatureExtractor
from alloskin.io.readers import MDAnalysisReader
from alloskin.pipeline.builder import DatasetBuilder


@dataclass
class DescriptorBuildResult:
    active_features: FeatureDict
    inactive_features: FeatureDict
    n_frames_active: int
    n_frames_inactive: int
    residue_keys: List[str]
    residue_mapping: Dict[str, str]


class DescriptorPreprocessor:
    """
    Wraps the DatasetBuilder to extract the phi/psi/chi1 descriptors once,
    without persisting the original trajectory files.
    """

    def __init__(self, residue_selections: Optional[Dict[str, str]] = None):
        reader = MDAnalysisReader()
        extractor = FeatureExtractor(residue_selections)
        self.builder = DatasetBuilder(reader, extractor)

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

        if not common_keys:
            raise ValueError("No overlapping residues found between active and inactive states.")

        active_common = {k: features_active[k] for k in common_keys}
        inactive_common = {k: features_inactive[k] for k in common_keys}

        return DescriptorBuildResult(
            active_features=active_common,
            inactive_features=inactive_common,
            n_frames_active=n_frames_active,
            n_frames_inactive=n_frames_inactive,
            residue_keys=sorted(common_keys),
            residue_mapping=mapping,
        )

