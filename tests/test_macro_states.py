from backend.services.project_store import DescriptorState as BackendDescriptorState
from phase.services.project_store import SystemMetadata
from phase.services.state_utils import build_analysis_states


def test_build_analysis_states_accepts_backend_descriptor_state():
    state = BackendDescriptorState(
        state_id="A",
        name="Active",
        pdb_file="structures/A.pdb",
        descriptor_file="descriptors/A.npz",
        descriptor_metadata_file="descriptors/A.json",
        trajectory_file="traj.xtc",
        n_frames=10,
        stride=1,
        source_traj=None,
        slice_spec=None,
        residue_selection=None,
        residue_keys=[],
        residue_mapping={},
        metastable_labels_file=None,
        role=None,
    )
    system = SystemMetadata(
        system_id="sys",
        project_id="proj",
        name="System",
        description=None,
        created_at="2026-02-02T00:00:00",
        states={"A": state},
    )
    analysis_states = build_analysis_states(system)
    assert any(s.get("state_id") == "A" and s.get("kind") == "macro" for s in analysis_states)
