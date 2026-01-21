"""
Parses and expands residue selection strings with special wildcards.

In addition to the legacy JSON dictionary, this module now supports
newline-delimited selections. Each line is treated as a standalone
selection and can optionally end with the `[singles]` or `[pairs]`
wildcards described below.

Wildcard Syntax:
- `[singles]`: Expands a selection into its constituent single residues.
  Example: "resid 10 to 12 [singles]" -> {'res_10': 'resid 10', 'res_11': 'resid 11', ...}
- `[pairs]`: Expands a selection into all unique pairwise combinations of its residues.
  Example: "resid 10 to 12 [pairs]" -> {'res_10_11': 'resid 10 11', ...}
"""
import re
import itertools
from typing import Dict, List, Optional, Union

import MDAnalysis as mda

SelectionConfig = Optional[Union[Dict[str, str], List[str]]]


def _slugify_selection_label(selection: str, fallback: str) -> str:
    """Create a deterministic key for unnamed selections."""
    slug = re.sub(r"[^a-z0-9]+", "_", selection.lower()).strip("_")
    slug = slug[:48]  # keep keys readable
    return slug or fallback


def _get_residues_from_selection(universe: mda.Universe, selection_str: str) -> mda.ResidueGroup:
    """Helper to safely select residues from a universe."""
    try:
        return universe.select_atoms(selection_str).residues
    except Exception:
        # This can happen with invalid selection strings. The caller should handle it.
        return universe.select_atoms("resname NONEXISTENT").residues


def expand_selection_wildcards(
    universe: mda.Universe,
    selections_dict: Dict[str, str]
) -> Dict[str, str]:
    """
    Expands a dictionary of residue selections containing wildcards.

    Args:
        universe: An MDAnalysis Universe object to resolve selections against.
        selections_dict: The user-provided dictionary of selections, which may
                         contain `[singles]` or `[pairs]` wildcards.

    Returns:
        A new dictionary with the expanded selections.
    """
    expanded_selections = {}
    
    for key, sel_string in selections_dict.items():
        
        # --- Case 1: Handle [singles] wildcard ---
        if '[singles]' in sel_string:
            base_sel = sel_string.replace('[singles]', '').strip()
            residues = _get_residues_from_selection(universe, base_sel)
            
            if residues.n_residues == 0:
                print(f"  Warning: Wildcard selection '{sel_string}' for key '{key}' matched 0 residues. Skipping.")
                continue

            print(f"  Expanding [singles] for '{key}': found {residues.n_residues} residues.")
            for res in residues:
                new_key = f"res_{res.resid}"
                expanded_selections[new_key] = f"resid {res.resid}"

        # --- Case 2: Handle [pairs] wildcard ---
        elif '[pairs]' in sel_string:
            base_sel = sel_string.replace('[pairs]', '').strip()
            residues = _get_residues_from_selection(universe, base_sel)

            if residues.n_residues < 2:
                print(f"  Warning: Wildcard selection '{sel_string}' for key '{key}' matched < 2 residues. Cannot create pairs. Skipping.")
                continue
            
            print(f"  Expanding [pairs] for '{key}': found {residues.n_residues} residues, creating pairs.")
            # Use resids for stable, human-readable keys
            resids = [res.resid for res in residues]
            
            for r1, r2 in itertools.combinations(resids, 2):
                # Sort to ensure key is consistent (e.g., res_10_11 not res_11_10)
                key_resids = sorted([r1, r2])
                new_key = f"res_{key_resids[0]}_{key_resids[1]}"
                expanded_selections[new_key] = f"resid {r1} {r2}"

        # --- Case 3: No wildcard, add directly ---
        else:
            expanded_selections[key] = sel_string
    
    return expanded_selections


def _normalize_selections(
    raw_selections: SelectionConfig
) -> Dict[str, str]:
    """
    Converts supported user input formats into a normalized dictionary.
    Accepts dictionaries (legacy JSON) or iterables of strings
    (newline-delimited textarea).
    """
    if raw_selections is None:
        return {}

    if isinstance(raw_selections, dict):
        # Filter out empty strings but retain legacy keys
        return {
            k: v.strip()
            for k, v in raw_selections.items()
            if isinstance(v, str) and v.strip()
        }

    if isinstance(raw_selections, (list, tuple)):
        normalized: Dict[str, str] = {}
        for idx, raw in enumerate(raw_selections, start=1):
            if raw is None:
                continue
            selection = str(raw).strip()
            if not selection:
                continue
            base_key = _slugify_selection_label(selection, f"selection_{idx}")
            key = base_key
            suffix = 2
            while key in normalized:
                key = f"{base_key}_{suffix}"
                suffix += 1
            normalized[key] = selection
        return normalized

    raise TypeError(f"Unsupported residue selection type: {type(raw_selections)!r}")


def parse_and_expand_selections(
    universe: mda.Universe,
    config_selections: SelectionConfig
) -> Dict[str, str]:
    """
    Main entry point to generate the final selection mapping.
    If no selections are provided, it generates them for all protein residues.
    If selections are provided, it expands any wildcards.
    """
    normalized = _normalize_selections(config_selections)

    if not normalized:
        print("  No residue selections provided. Generating selections for all protein residues...")
        protein_residues = universe.select_atoms('protein').residues
        selections_to_process = {
            f"res_{res.resid}": f"resid {res.resid}" for res in protein_residues
        }
        print(f"  Generated {len(selections_to_process)} selections to check against inactive state.")
    else:
        print("  Expanding residue selections with wildcards (if any)...")
        selections_to_process = expand_selection_wildcards(universe, normalized)

    return selections_to_process
