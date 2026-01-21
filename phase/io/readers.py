"""
Handles reading molecular dynamics trajectories from disk.
This class is deliberately simple; it only loads a Universe.
Slicing is handled at the analysis level (in the extractor).
"""

import MDAnalysis as mda
from abc import ABC, abstractmethod
from typing import Any

class AbstractTrajectoryReader(ABC):
    """
    Abstract interface for reading trajectories.
    Allows swapping mdtraj for MDAnalysis, etc.
    """
    @abstractmethod
    def load_trajectory(self, trajectory_file: str, topology_file: str) -> Any:
        pass

class MDAnalysisReader(AbstractTrajectoryReader):
    """Concrete implementation using MDAnalysis."""
    
    def load_trajectory(self, trajectory_file: str, topology_file: str) -> mda.Universe:
        """
        Loads trajectory and topology files into an MDAnalysis Universe.
        
        Args:
            trajectory_file: Path to the trajectory file (e.g., .xtc, .dcd).
            topology_file: Path to the topology file (e.g., .pdb, .gro).
            
        Returns:
            mda.Universe: A Universe object.
        """
        print(f"Loading trajectory: {trajectory_file} with topology {topology_file}...")
        try:
            # MDAnalysis Universe creation: (topology, trajectory)
            universe = mda.Universe(topology_file, trajectory_file)
            print(f"Universe loaded: {len(universe.trajectory)} total frames, {len(universe.atoms)} atoms.")
            return universe
        except Exception as e:
            print(f"Error loading trajectory {trajectory_file} with topology {topology_file}: {e}")
            raise