from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from backend.api.v1.common import project_store


ANALYSIS_METADATA_FILENAME = "analysis_metadata.json"

_DIRECT_SAMPLE_ANALYSIS_TYPES = {
    "md_vs_sample",
    "model_energy",
    "delta_eval",
    "delta_transition",
    "gibbs_relaxation",
    "ligand_completion",
    "lambda_sweep",
}

_MULTI_SAMPLE_ANALYSIS_TYPES = {
    "delta_commitment",
    "delta_js",
}

_SINGLE_STATE_REF_KEYS = {
    "state_id",
    "state_a_id",
    "state_b_id",
    "start_state_id",
    "macro_state_id",
    "metastable_id",
    "metastable_a_id",
    "metastable_b_id",
}

_MULTI_STATE_REF_KEYS = {
    "state_ids",
    "metastable_ids",
    "selected_state_ids",
    "selected_metastable_ids",
    "contact_state_ids",
    "fit_state_ids",
    "fit_sample_state_ids",
}


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if "," in raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [raw]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _extract_sample_refs(meta: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    direct: set[str] = set()
    essential: set[str] = set()
    optional_many: set[str] = set()

    for key in (
        "sample_id",
        "md_sample_id",
        "active_md_sample_id",
        "inactive_md_sample_id",
        "pas_md_sample_id",
        "start_sample_id",
        "reference_sample_id_a",
        "reference_sample_id_b",
        "constraint_delta_js_sample_id",
    ):
        values = _coerce_str_list(meta.get(key))
        direct.update(values)
        essential.update(values)

    for key in ("md_sample_ids", "reference_sample_ids_a", "reference_sample_ids_b"):
        values = _coerce_str_list(meta.get(key))
        direct.update(values)
        essential.update(values)

    summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
    optional_many.update(_coerce_str_list(summary.get("sample_ids")))

    potts_overlay = meta.get("potts_overlay") if isinstance(meta.get("potts_overlay"), dict) else {}
    optional_many.update(_coerce_str_list(potts_overlay.get("sample_ids_a")))
    optional_many.update(_coerce_str_list(potts_overlay.get("sample_ids_b")))

    return direct | optional_many, essential, optional_many


def _extract_model_refs(meta: dict[str, Any]) -> tuple[set[str], set[str]]:
    refs: set[str] = set()
    essential: set[str] = set()
    for key in ("model_id", "model_a_id", "model_b_id"):
        values = _coerce_str_list(meta.get(key))
        refs.update(values)
        essential.update(values)
    return refs, essential


def _extract_state_refs(meta: Any) -> set[str]:
    refs: set[str] = set()

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_str = str(key)
                if key_str in _SINGLE_STATE_REF_KEYS or key_str in _MULTI_STATE_REF_KEYS:
                    refs.update(_coerce_str_list(value))
                    continue
                if key_str == "states" and isinstance(value, dict):
                    for state_ref in value.values():
                        if isinstance(state_ref, dict):
                            refs.update(_coerce_str_list(state_ref.get("id")))
                    continue
                _walk(value)
            return
        if isinstance(node, (list, tuple, set)):
            for item in node:
                _walk(item)

    _walk(meta)
    return {ref for ref in refs if ref}


def _valid_state_ids(system_meta: Any) -> set[str]:
    valid: set[str] = {str(sid) for sid in (getattr(system_meta, "states", {}) or {}).keys()}
    for meta in getattr(system_meta, "metastable_states", []) or []:
        meta_id = None
        if isinstance(meta, dict):
            meta_id = meta.get("metastable_id") or meta.get("id")
        if meta_id:
            valid.add(str(meta_id))
    return valid


def _analysis_is_orphan(
    meta: dict[str, Any],
    sample_ids: set[str],
    model_ids: set[str],
    valid_state_ids: set[str] | None = None,
) -> bool:
    analysis_type = str(meta.get("analysis_type") or "").strip().lower()
    sample_refs, essential_sample_refs, optional_sample_refs = _extract_sample_refs(meta)
    _all_model_refs, essential_model_refs = _extract_model_refs(meta)
    if valid_state_ids is not None:
        state_refs = _extract_state_refs(meta)
        if state_refs and any(ref not in valid_state_ids for ref in state_refs):
            return True

    if essential_model_refs and any(ref not in model_ids for ref in essential_model_refs):
        return True
    if essential_sample_refs and any(ref not in sample_ids for ref in essential_sample_refs):
        return True

    if analysis_type in _DIRECT_SAMPLE_ANALYSIS_TYPES:
        if sample_refs and any(ref not in sample_ids for ref in sample_refs):
            return True
        return False

    if analysis_type in _MULTI_SAMPLE_ANALYSIS_TYPES:
        if optional_sample_refs and not any(ref in sample_ids for ref in optional_sample_refs):
            return True
        return False

    return False


def cleanup_orphan_cluster_analyses(project_id: str, system_id: str, cluster_id: str) -> int:
    cluster_dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    analyses_root = cluster_dirs["cluster_dir"] / "analyses"
    if not analyses_root.exists():
        return 0
    try:
        system_meta = project_store.get_system(project_id, system_id)
        valid_state_ids = _valid_state_ids(system_meta)
    except Exception:
        valid_state_ids = set()

    sample_ids = {
        str(entry.get("sample_id"))
        for entry in project_store.list_samples(project_id, system_id, cluster_id)
        if isinstance(entry, dict) and entry.get("sample_id")
    }
    model_ids = {
        str(entry.get("model_id"))
        for entry in project_store.list_potts_models(project_id, system_id, cluster_id)
        if isinstance(entry, dict) and entry.get("model_id")
    }

    removed = 0
    for kind_dir in sorted((p for p in analyses_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for analysis_dir in sorted((p for p in kind_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
            meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
            npz_path = analysis_dir / "analysis.npz"
            orphan = False
            if not meta_path.exists() or not npz_path.exists():
                orphan = True
            else:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    orphan = True
                else:
                    orphan = _analysis_is_orphan(
                        meta,
                        sample_ids=sample_ids,
                        model_ids=model_ids,
                        valid_state_ids=valid_state_ids,
                    )
            if not orphan:
                continue
            shutil.rmtree(analysis_dir, ignore_errors=True)
            removed += 1

        try:
            if not any(kind_dir.iterdir()):
                kind_dir.rmdir()
        except Exception:
            pass
    return removed


def remove_md_samples_for_states(
    project_id: str,
    system_id: str,
    state_ids: Iterable[str],
) -> dict[str, Any]:
    """Remove md_eval sample folders associated with deleted macro states."""
    target_ids = {str(state_id).strip() for state_id in state_ids if str(state_id).strip()}
    removed_sample_ids: list[str] = []
    affected_cluster_ids: list[str] = []
    if not target_ids:
        return {
            "md_samples_removed": 0,
            "removed_md_sample_ids": removed_sample_ids,
            "affected_cluster_ids": affected_cluster_ids,
        }

    try:
        entries = project_store.list_cluster_entries(project_id, system_id)
    except Exception:
        entries = []
    for entry in entries:
        cluster_id = str(entry.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        matched_ids: list[str] = []
        for sample in entry.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            if str(sample.get("type") or "").strip().lower() != "md_eval":
                continue
            state_refs = {
                str(sample.get(key) or "").strip()
                for key in ("state_id", "macro_state_id", "metastable_id")
            }
            state_refs.discard("")
            if not (state_refs & target_ids):
                continue
            sample_id = str(sample.get("sample_id") or "").strip()
            if sample_id:
                matched_ids.append(sample_id)
        if not matched_ids:
            continue
        cluster_dir = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)["cluster_dir"]
        for sample_id in matched_ids:
            shutil.rmtree(cluster_dir / "samples" / sample_id, ignore_errors=True)
            removed_sample_ids.append(sample_id)
        affected_cluster_ids.append(cluster_id)

    return {
        "md_samples_removed": len(removed_sample_ids),
        "removed_md_sample_ids": removed_sample_ids,
        "affected_cluster_ids": affected_cluster_ids,
    }


def cleanup_results_for_samples(project_id: str, system_id: str, sample_ids: Iterable[str]) -> int:
    """Remove stored job results that directly reference deleted samples."""
    target_ids = {str(sample_id).strip() for sample_id in sample_ids if str(sample_id).strip()}
    if not target_ids:
        return 0
    system_dir = project_store.resolve_path(project_id, system_id, "")
    removed = 0

    def references_target(node: Any) -> bool:
        if isinstance(node, dict):
            refs, _essential, _optional = _extract_sample_refs(node)
            if refs & target_ids:
                return True
            for key, value in node.items():
                key_text = str(key).lower()
                if key_text.endswith("sample_id") or key_text.endswith("sample_ids"):
                    if set(_coerce_str_list(value)) & target_ids:
                        return True
                if references_target(value):
                    return True
            return False
        if isinstance(node, (list, tuple, set)):
            return any(references_target(item) for item in node)
        if isinstance(node, str):
            return any(sample_id in Path(node).parts for sample_id in target_ids)
        return False

    for result_file in _iter_result_files_for_system(project_id, system_id):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not references_target(payload):
            continue
        try:
            results_payload = payload.get("results") if isinstance(payload, dict) else {}
            _remove_results_dir((results_payload or {}).get("results_dir"), system_dir=system_dir)
            result_file.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    return removed


def _iter_result_files_for_system(project_id: str, system_id: str) -> list[Path]:
    jobs_dir = project_store.resolve_path(project_id, system_id, "") / "results" / "jobs"
    if not jobs_dir.exists():
        return []
    return sorted((p for p in jobs_dir.glob("*.json") if p.is_file()), key=lambda p: p.name)


def _remove_results_dir(path_value: str | None, *, system_dir: Path) -> None:
    if not isinstance(path_value, str) or not path_value:
        return
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = project_store.base_dir.parent / path_value
    candidate = candidate.resolve()
    try:
        candidate.relative_to(system_dir)
    except ValueError:
        results_root = (system_dir / "results").resolve()
        try:
            candidate.relative_to(results_root)
        except ValueError:
            return
    if candidate.exists() and candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)


def cleanup_state_linked_results(project_id: str, system_id: str) -> int:
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except Exception:
        return 0

    valid_state_ids = _valid_state_ids(system_meta)
    system_dir = project_store.resolve_path(project_id, system_id, "")
    removed = 0
    for result_file in _iter_result_files_for_system(project_id, system_id):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        state_refs = _extract_state_refs(payload)
        if not state_refs or all(ref in valid_state_ids for ref in state_refs):
            continue
        try:
            results_payload = payload.get("results") if isinstance(payload, dict) else {}
            _remove_results_dir((results_payload or {}).get("results_dir"), system_dir=system_dir)
            result_file.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue

    jobs_dir = system_dir / "results" / "jobs"
    try:
        if jobs_dir.exists() and not any(jobs_dir.iterdir()):
            jobs_dir.rmdir()
    except Exception:
        pass
    try:
        results_dir = system_dir / "results"
        if results_dir.exists() and not any(results_dir.iterdir()):
            results_dir.rmdir()
    except Exception:
        pass
    return removed


def cleanup_state_linked_artifacts(project_id: str, system_id: str) -> dict[str, int]:
    cluster_removed = 0
    try:
        entries = project_store.list_cluster_entries(project_id, system_id)
    except Exception:
        entries = []
    for entry in entries:
        cluster_id = str(entry.get("cluster_id") or "").strip()
        if not cluster_id:
            continue
        cluster_removed += cleanup_orphan_cluster_analyses(project_id, system_id, cluster_id)
    result_removed = cleanup_state_linked_results(project_id, system_id)
    return {
        "cluster_analyses_removed": cluster_removed,
        "results_removed": result_removed,
    }


def cleanup_orphan_analyses_all_systems() -> int:
    removed = 0
    for project in project_store.list_projects():
        project_id = str(project.project_id)
        for system in project_store.list_systems(project_id):
            system_id = str(system.system_id)
            for entry in project_store.list_cluster_entries(project_id, system_id):
                cluster_id = str(entry.get("cluster_id") or "").strip()
                if not cluster_id:
                    continue
                removed += cleanup_orphan_cluster_analyses(project_id, system_id, cluster_id)
    return removed


def cleanup_state_linked_artifacts_all_systems() -> dict[str, int]:
    totals = {
        "cluster_analyses_removed": 0,
        "results_removed": 0,
    }
    for project in project_store.list_projects():
        project_id = str(project.project_id)
        for system in project_store.list_systems(project_id):
            system_id = str(system.system_id)
            removed = cleanup_state_linked_artifacts(project_id, system_id)
            for key, value in removed.items():
                totals[key] = totals.get(key, 0) + int(value or 0)
    return totals
