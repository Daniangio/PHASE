from __future__ import annotations

from typing import Optional

from backend.services.project_store import SelectionInput


def build_residue_selection_config(
    *,
    base_selections: Optional[SelectionInput],
    residue_filter: Optional[str],
) -> Optional[SelectionInput]:
    """
    If a residue_filter is provided, build a selection list that intersects
    protein residues with the filter and expands to singles. Otherwise, fall
    back to the base selections (or None for default protein selection).
    """
    if residue_filter is None:
        return base_selections
    selection = residue_filter.strip()
    if not selection:
        return base_selections
    combined = f"protein and ({selection})"
    return [f"{combined} [singles]"]
