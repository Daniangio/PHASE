from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from phase.potts.potts_model import PottsModel, add_potts_models, load_potts_model
from phase.potts.qubo import decode_onehot, encode_onehot, potts_to_qubo_onehot
from phase.potts.sample_io import SAMPLE_NPZ_FILENAME, load_sample_npz, save_sample_npz
from phase.potts.sampling import (
    _ProgressCounter,
    gibbs_sample_potts,
    make_beta_ladder,
    replica_exchange_gibbs_potts,
    sa_sample_qubo_neal,
)


@dataclass(frozen=True)
class SamplingResult:
    sample_path: Path
    n_samples: int
    n_residues: int
    sa_diagnostics: Optional[Dict[str, object]] = None


def _normalize_sa_restart(value: object) -> str:
    """
    Normalize SA restart mode across legacy entry-points.

    Sampling supports:
    - previous: correlated chain where each sample warm-starts from the previous sample
    - md: correlated chain where each sample warm-starts from a fresh MD frame
    - independent: independent SA reads/chains

    Older UI/CLI variants used "prev-topk"/"prev-uniform" (from the old multi-schedule pipeline);
    we map those to "previous" here for compatibility.
    """
    raw = "" if value is None else str(value)
    s = raw.strip().lower()
    if s in {"previous", "prev", "chain", "prev-topk", "prev-uniform"}:
        return "previous"
    if s in {"md", "md-frame", "md-random", "md_random"}:
        return "md"
    if s in {"independent", "indep", "iid", "random", "rand"}:
        return "independent"
    return s


def _normalize_model_paths(model_npz: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in model_npz or []:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        if "," in s:
            out.extend([p.strip() for p in s.split(",") if p.strip()])
        else:
            out.append(s)
    return out


def _load_combined_model(model_paths: Sequence[str]) -> PottsModel:
    paths = _normalize_model_paths(model_paths)
    if not paths:
        raise ValueError("No --model-npz provided (sampling requires an existing Potts model).")
    model = load_potts_model(paths[0])
    for p in paths[1:]:
        model = add_potts_models(model, load_potts_model(p))
    return model


def _sample_labels_uniform(K_list: Sequence[int], n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n_res = len(K_list)
    out = np.zeros((n_samples, n_res), dtype=int)
    for r, k in enumerate(K_list):
        out[:, r] = rng.integers(0, int(k), size=n_samples)
    return out


def _sample_labels_from_fields(
    model: PottsModel,
    *,
    beta: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_res = len(model.h)
    out = np.zeros((n_samples, n_res), dtype=int)
    for r, hr in enumerate(model.h):
        hr = np.asarray(hr, dtype=float)
        if hr.size == 0 or not np.all(np.isfinite(hr)):
            out[:, r] = rng.integers(0, max(1, hr.size), size=n_samples)
            continue
        logits = -float(beta) * hr
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        total = float(np.sum(probs))
        if total <= 0 or not np.isfinite(total):
            out[:, r] = rng.integers(0, hr.shape[0], size=n_samples)
            continue
        probs = probs / total
        out[:, r] = rng.choice(hr.shape[0], size=n_samples, p=probs)
    return out


def _build_sa_initial_labels(
    *,
    mode: str,
    md_labels: np.ndarray,
    model: PottsModel,
    beta: float,
    n_reads: int,
    md_frame: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mode = (mode or "md").lower()
    if mode in {"md", "md-frame"}:
        if md_labels is None or md_labels.size == 0:
            if mode == "md-frame":
                raise ValueError("SA init set to md-frame, but MD labels are unavailable.")
            return _sample_labels_from_fields(model, beta=beta, n_samples=n_reads, rng=rng), np.full(n_reads, -1, dtype=np.int64)
        if mode == "md-frame":
            if md_frame < 0:
                raise ValueError("--sa-init md-frame requires --sa-init-md-frame >= 0.")
            if md_frame >= md_labels.shape[0]:
                raise ValueError(f"--sa-init-md-frame {md_frame} out of range (0..{md_labels.shape[0]-1}).")
            return np.repeat(md_labels[md_frame : md_frame + 1], n_reads, axis=0), np.full(n_reads, md_frame, dtype=np.int64)
        idx = rng.integers(0, md_labels.shape[0], size=n_reads)
        return md_labels[idx], np.asarray(idx, dtype=np.int64)
    if mode in {"random-h", "h"}:
        return _sample_labels_from_fields(model, beta=beta, n_samples=n_reads, rng=rng), np.full(n_reads, -1, dtype=np.int64)
    if mode in {"random-uniform", "uniform"}:
        return _sample_labels_uniform(model.K_list(), n_reads, rng), np.full(n_reads, -1, dtype=np.int64)
    raise ValueError(f"Unknown sa-init mode: {mode}")


def _parse_float_list(raw: str) -> List[float]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return [float(p) for p in parts]


def _normalize_sa_schedule_type(value: object) -> str:
    raw = "" if value is None else str(value)
    s = raw.strip().lower()
    if s in {"geom", "geometric"}:
        return "geometric"
    if s in {"lin", "linear"}:
        return "linear"
    if s == "custom":
        return "custom"
    return s


def _parse_sa_custom_schedule(value: object) -> List[float]:
    if value is None:
        return []
    if isinstance(value, str):
        return _parse_float_list(value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=float).ravel()
        return [float(v) for v in arr.tolist()]
    if isinstance(value, Sequence):
        out: List[float] = []
        for item in value:
            out.append(float(item))
        return out
    return [float(value)]


def _run_gibbs_chain_worker(payload: dict[str, object]) -> dict[str, object]:
    labels = gibbs_sample_potts(
        payload["model"],  # type: ignore[arg-type]
        beta=float(payload["beta"]),
        n_samples=int(payload["n_samples"]),
        burn_in=int(payload["burn_in"]),
        thinning=int(payload["thinning"]),
        seed=int(payload["seed"]),
        progress=bool(payload.get("progress", False)),
        progress_mode=str(payload.get("progress_mode", "samples")),
        progress_desc=str(payload.get("progress_desc", "Gibbs samples")),
        progress_position=payload.get("progress_position"),  # type: ignore[arg-type]
    )
    return {"labels": labels}


def _run_rex_chain_worker(payload: dict[str, object]) -> dict[str, object]:
    return replica_exchange_gibbs_potts(
        payload["model"],  # type: ignore[arg-type]
        betas=payload["betas"],  # type: ignore[arg-type]
        sweeps_per_round=int(payload["sweeps_per_round"]),
        n_rounds=int(payload["n_rounds"]),
        burn_in_rounds=int(payload["burn_in_rounds"]),
        thinning_rounds=int(payload["thinning_rounds"]),
        seed=int(payload["seed"]),
        progress=bool(payload.get("progress", False)),
        progress_callback=None,
        progress_every=max(1, int(payload.get("progress_every", 1))),
        max_workers=payload.get("max_workers"),
        progress_desc=payload.get("progress_desc"),  # type: ignore[arg-type]
        progress_position=payload.get("progress_position"),  # type: ignore[arg-type]
        progress_mode=str(payload.get("progress_mode", "samples")),
    )

def _filter_md_pool_for_states(
    labels: np.ndarray,
    frame_state_ids: np.ndarray | None,
    frame_indices: np.ndarray,
    *,
    state_ids: Sequence[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if labels is None:
        return labels, frame_indices
    if not state_ids:
        return labels, frame_indices
    if frame_state_ids is None:
        raise ValueError("MD frame_state_ids missing in cluster NPZ; cannot filter by --sa-md-state-ids.")
    ids = [str(s).strip() for s in state_ids if str(s).strip()]
    if not ids:
        return labels, frame_indices
    frame_ids = np.asarray(frame_state_ids).astype(str)
    mask = np.isin(frame_ids, ids)
    if not np.any(mask):
        raise ValueError(f"No MD frames matched sa_md_state_ids={ids}.")
    return np.asarray(labels)[mask], np.asarray(frame_indices, dtype=np.int64)[mask]


def _load_sa_md_labels(payload: dict[str, object]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    sample_path = str(payload.get("sa_md_sample_npz") or "").strip()
    sample_id = str(payload.get("sa_md_sample_id") or "").strip()
    if not sample_path:
        raise ValueError("SA sampling requires sa_md_sample_npz.")
    sample = load_sample_npz(sample_path)
    labels = np.asarray(sample.labels, dtype=np.int32)
    if labels.ndim != 2 or labels.shape[0] == 0:
        ident = sample_id or sample_path
        raise ValueError(f"Selected MD sample has no usable labels: {ident}")
    frame_state_ids = (
        np.asarray(sample.frame_state_ids, dtype=str)
        if sample.frame_state_ids is not None and sample.frame_state_ids.shape[0] == labels.shape[0]
        else None
    )
    frame_indices = (
        np.asarray(sample.frame_indices, dtype=np.int64)
        if sample.frame_indices is not None and sample.frame_indices.shape[0] == labels.shape[0]
        else np.arange(labels.shape[0], dtype=np.int64)
    )
    return labels, frame_state_ids, frame_indices


def _parse_str_list(raw: str) -> List[str]:
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
    return [str(p) for p in parts]


def _sa_decode_labels(
    Z: np.ndarray,
    qubo,
    *,
    repair: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repair_mode = None if str(repair) == "none" else str(repair)
    labels = np.zeros((Z.shape[0], len(qubo.var_slices)), dtype=np.int32)
    valid_counts = np.zeros(Z.shape[0], dtype=np.int32)
    for i in range(Z.shape[0]):
        x, valid = decode_onehot(Z[i], qubo, repair=repair_mode)
        labels[i] = x
        valid_counts[i] = int(valid.sum())
    invalid_mask = valid_counts != int(labels.shape[1])
    return labels, invalid_mask.astype(bool), valid_counts


def _sa_project_labels_for_restart(z: np.ndarray, qubo) -> np.ndarray:
    """
    Project a raw QUBO bitstring to one valid label per residue for chain restarts.

    This is deliberately separate from the user-facing `repair` mode:
    even if we keep invalid samples marked as invalid in the saved outputs,
    a correlated SA chain needs a concrete next label assignment. Using argmax
    within each residue slice is more stable than silently resetting invalid
    residues to state 0.
    """
    z = np.asarray(z, dtype=int)
    x = np.zeros(len(qubo.var_slices), dtype=np.int32)
    for r, sl in enumerate(qubo.var_slices):
        x[r] = int(np.argmax(z[sl]))
    return x


def _run_sa_independent_worker(payload: dict[str, object]) -> dict[str, object]:
    model = _load_combined_model(payload["model_npz"])  # type: ignore[arg-type]
    beta = float(payload["beta"])
    penalty_safety = float(payload["penalty_safety"])
    n_reads = int(payload["n_reads"])
    sweeps = int(payload["sweeps"])
    seed = int(payload["seed"])
    beta_range = payload.get("beta_range")  # type: ignore[assignment]
    beta_schedule_type = _normalize_sa_schedule_type(payload.get("sa_schedule_type", "geometric"))
    beta_schedule = _parse_sa_custom_schedule(payload.get("sa_custom_beta_schedule"))
    sweeps_per_beta = int(payload.get("sa_num_sweeps_per_beta", 2))
    randomize_order = bool(payload.get("sa_randomize_order", True))
    acceptance = str(payload.get("sa_acceptance_criteria", "Metropolis"))
    sa_init = str(payload.get("sa_init", "md"))
    sa_init_md_frame = int(payload.get("sa_init_md_frame", -1))
    repair = str(payload.get("repair", "none"))

    md_labels, md_frame_state_ids, md_frame_indices = _load_sa_md_labels(payload)
    md_state_ids = _parse_str_list(str(payload.get("sa_md_state_ids", "")))
    md_labels, md_frame_indices = _filter_md_pool_for_states(
        md_labels, md_frame_state_ids, md_frame_indices, state_ids=md_state_ids
    )

    qubo = potts_to_qubo_onehot(model, beta=beta, penalty_safety=penalty_safety)
    init_rng = np.random.default_rng(seed + 1000)
    init_labels, init_pool_indices = _build_sa_initial_labels(
        mode=sa_init,
        md_labels=md_labels,
        model=model,
        beta=beta,
        n_reads=n_reads,
        md_frame=sa_init_md_frame,
        rng=init_rng,
    )
    init_frame_indices = np.full(n_reads, -1, dtype=np.int64)
    has_md_start = init_pool_indices >= 0
    init_frame_indices[has_md_start] = md_frame_indices[init_pool_indices[has_md_start]]
    init_states = encode_onehot(init_labels, qubo) if init_labels is not None and init_labels.size else None

    Z = sa_sample_qubo_neal(
        qubo,
        n_reads=n_reads,
        sweeps=sweeps,
        seed=seed,
        progress=bool(payload.get("progress", False)),
        beta_range=beta_range,  # type: ignore[arg-type]
        beta_schedule_type=beta_schedule_type,
        beta_schedule=beta_schedule or None,
        num_sweeps_per_beta=sweeps_per_beta,
        randomize_order=randomize_order,
        proposal_acceptance_criteria=acceptance,
        initial_states=init_states,
    )
    labels, invalid_mask, valid_counts = _sa_decode_labels(Z, qubo, repair=repair)
    return {
        "labels": labels,
        "invalid_mask": invalid_mask,
        "valid_counts": valid_counts,
        "sa_initial_labels": np.asarray(init_labels, dtype=np.int32),
        "sa_initial_md_frame_indices": init_frame_indices,
    }


def _run_sa_chain_worker(payload: dict[str, object]) -> dict[str, object]:
    """
    Sequential SA chain: each sample starts from either the previous sample ("previous")
    or a fresh random MD frame ("md").
    """
    try:
        import dimod  # type: ignore
        import neal  # type: ignore
    except Exception as e:
        raise RuntimeError("neal/dimod are not installed.") from e

    model = _load_combined_model(payload["model_npz"])  # type: ignore[arg-type]
    beta = float(payload["beta"])
    penalty_safety = float(payload["penalty_safety"])
    n_samples = int(payload["n_samples"])
    sweeps = int(payload["sweeps"])
    seed = int(payload["seed"])
    beta_range = payload.get("beta_range")  # type: ignore[assignment]
    beta_schedule_type = _normalize_sa_schedule_type(payload.get("sa_schedule_type", "geometric"))
    beta_schedule = _parse_sa_custom_schedule(payload.get("sa_custom_beta_schedule"))
    sweeps_per_beta = int(payload.get("sa_num_sweeps_per_beta", 2))
    randomize_order = bool(payload.get("sa_randomize_order", True))
    acceptance = str(payload.get("sa_acceptance_criteria", "Metropolis"))
    sa_init = str(payload.get("sa_init", "md"))
    sa_init_md_frame = int(payload.get("sa_init_md_frame", -1))
    sa_restart = _normalize_sa_restart(payload.get("sa_restart", "independent"))
    repair = str(payload.get("repair", "none"))

    if sa_restart not in {"previous", "md"}:
        raise ValueError("--sa-restart must be one of: previous, md (for chain mode).")

    md_labels, md_frame_state_ids, md_frame_indices = _load_sa_md_labels(payload)
    md_state_ids = _parse_str_list(str(payload.get("sa_md_state_ids", "")))
    md_labels, md_frame_indices = _filter_md_pool_for_states(
        md_labels, md_frame_state_ids, md_frame_indices, state_ids=md_state_ids
    )

    # Build QUBO and corresponding BQM once.
    qubo = potts_to_qubo_onehot(model, beta=beta, penalty_safety=penalty_safety)
    linear = {i: float(qubo.a[i]) for i in range(qubo.num_vars())}
    quadratic = {k: float(v) for k, v in qubo.Q.items()}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, float(qubo.const), dimod.BINARY)
    sampler = neal.SimulatedAnnealingSampler()

    init_rng = np.random.default_rng(seed + 1000)
    next_init_rows, next_pool_indices = _build_sa_initial_labels(
        mode=sa_init,
        md_labels=md_labels,
        model=model,
        beta=beta,
        n_reads=1,
        md_frame=sa_init_md_frame,
        rng=init_rng,
    )
    next_init = next_init_rows[0]
    next_frame_index = int(md_frame_indices[next_pool_indices[0]]) if int(next_pool_indices[0]) >= 0 else -1

    repair_mode = None if str(repair) == "none" else str(repair)
    labels = np.zeros((n_samples, len(qubo.var_slices)), dtype=np.int32)
    valid_counts = np.zeros(n_samples, dtype=np.int32)
    invalid_mask = np.zeros(n_samples, dtype=bool)
    initial_labels = np.zeros((n_samples, len(qubo.var_slices)), dtype=np.int32)
    initial_md_frame_indices = np.full(n_samples, -1, dtype=np.int64)
    sample_counter = _ProgressCounter(
        n_samples,
        str(payload.get("progress_desc") or "SA samples"),
        bool(payload.get("progress", False)),
        position=payload.get("progress_position"),  # type: ignore[arg-type]
    )

    try:
        for i in range(n_samples):
            initial_labels[i] = next_init
            initial_md_frame_indices[i] = next_frame_index
            init_state = encode_onehot(next_init, qubo)
            init = np.asarray(init_state, dtype=np.int8)[None, :]
            init_min = int(init.min()) if init.size else 0
            init_max = int(init.max()) if init.size else 0
            if init_min >= 0 and init_max <= 1:
                init = (init * 2 - 1).astype(np.int8, copy=False)
            elif init_min < -1 or init_max > 1:
                raise ValueError("initial state must be binary (0/1) or spin (-1/1).")
            init = np.ascontiguousarray(init, dtype=np.int8)

            kwargs: Dict[str, object] = {
                "num_reads": 1,
                "seed": int(seed) + int(i),
                "initial_states": init,
                "num_sweeps_per_beta": int(sweeps_per_beta),
                "randomize_order": bool(randomize_order),
                "proposal_acceptance_criteria": str(acceptance),
            }
            if beta_schedule:
                kwargs["beta_schedule_type"] = "custom"
                kwargs["beta_schedule"] = np.asarray(beta_schedule, dtype=float)
            else:
                kwargs["num_sweeps"] = sweeps
                kwargs["beta_schedule_type"] = str(beta_schedule_type)
            if beta_range is not None and not beta_schedule:
                kwargs["beta_range"] = beta_range  # type: ignore[assignment]

            def _sample_with_kwargs(sample_kwargs: Dict[str, object]):
                return sampler.sample(bqm, **sample_kwargs)

            try:
                ss = _sample_with_kwargs(kwargs)
            except TypeError:
                # Some neal versions require initial_states as a list[dict]
                init_list = [{j: int(init[0, j]) for j in range(qubo.num_vars())}]
                retry = dict(kwargs)
                retry["initial_states"] = init_list
                try:
                    ss = _sample_with_kwargs(retry)
                except Exception:
                    # Fall back to random init (best effort).
                    fallback = dict(kwargs)
                    fallback.pop("initial_states", None)
                    ss = _sample_with_kwargs(fallback)

            sample = next(iter(ss.samples()))
            z = np.zeros(qubo.num_vars(), dtype=int)
            for j in range(qubo.num_vars()):
                z[j] = int(sample[j])
            x, valid = decode_onehot(z, qubo, repair=repair_mode)
            labels[i] = x
            vc = int(valid.sum())
            valid_counts[i] = vc
            invalid_mask[i] = vc != int(labels.shape[1])

            if sa_restart == "previous":
                next_init = _sa_project_labels_for_restart(z, qubo)
                next_frame_index = -1
            else:
                # fresh MD init for each sample
                next_rows, next_indices = _build_sa_initial_labels(
                    mode="md",
                    md_labels=md_labels,
                    model=model,
                    beta=beta,
                    n_reads=1,
                    md_frame=-1,
                    rng=init_rng,
                )
                next_init = next_rows[0]
                next_frame_index = int(md_frame_indices[next_indices[0]]) if int(next_indices[0]) >= 0 else -1
            sample_counter.update(1)
    finally:
        sample_counter.close()

    return {
        "labels": labels,
        "invalid_mask": invalid_mask,
        "valid_counts": valid_counts,
        "sa_initial_labels": initial_labels,
        "sa_initial_md_frame_indices": initial_md_frame_indices,
    }


def run_sampling(
    *,
    cluster_npz: str,
    sa_md_sample_npz: str | None = None,
    results_dir: str | Path,
    model_npz: Sequence[str],
    sampling_method: str,
    beta: float,
    seed: int,
    progress: bool = False,
    # gibbs
    gibbs_method: str = "single",
    gibbs_samples: int = 500,
    gibbs_burnin: int = 50,
    gibbs_thin: int = 2,
    gibbs_chains: int = 1,
    # rex
    rex_betas: str = "",
    rex_n_replicas: int = 8,
    rex_beta_min: float = 0.2,
    rex_beta_max: float = 1.0,
    rex_spacing: str = "geom",
    rex_rounds: int = 2000,
    rex_burnin_rounds: int = 50,
    rex_sweeps_per_round: int = 2,
    rex_thin_rounds: int = 1,
    rex_chains: int = 1,
    # sa
    sa_reads: int = 2000,
    sa_chains: int = 1,
    sa_sweeps: int = 2000,
    sa_beta_hot: float = 0.01,
    sa_beta_cold: float = 2.0,
    sa_schedule_type: str = "geometric",
    sa_custom_beta_schedule: Sequence[float] | str | None = None,
    sa_num_sweeps_per_beta: int = 2,
    sa_randomize_order: bool = True,
    sa_acceptance_criteria: str = "Metropolis",
    sa_init: str = "md",
    sa_init_md_frame: int = -1,
    sa_restart: str = "independent",
    sa_restart_topk: int = 200,
    sa_md_sample_id: str = "",
    sa_md_state_ids: str = "",
    penalty_safety: float = 4.0,
    repair: str = "none",
    progress_callback: Callable[[str, int], None] | None = None,
) -> SamplingResult:
    from phase.potts.orchestration import run_sampling_local

    def report(message: str, current: int, total: int) -> None:
        if progress_callback is None or total <= 0:
            return
        pct = int(round(100.0 * float(current) / float(total)))
        progress_callback(message, pct)

    out = run_sampling_local(
        cluster_npz=cluster_npz,
        sa_md_sample_npz=sa_md_sample_npz,
        results_dir=results_dir,
        model_npz=model_npz,
        sampling_method=sampling_method,
        beta=beta,
        seed=seed,
        progress=progress,
        gibbs_method=gibbs_method,
        gibbs_samples=gibbs_samples,
        gibbs_burnin=gibbs_burnin,
        gibbs_thin=gibbs_thin,
        gibbs_chains=gibbs_chains,
        rex_betas=rex_betas,
        rex_n_replicas=rex_n_replicas,
        rex_beta_min=rex_beta_min,
        rex_beta_max=rex_beta_max,
        rex_spacing=rex_spacing,
        rex_rounds=rex_rounds,
        rex_burnin_rounds=rex_burnin_rounds,
        rex_sweeps_per_round=rex_sweeps_per_round,
        rex_thin_rounds=rex_thin_rounds,
        rex_chains=rex_chains,
        sa_reads=sa_reads,
        sa_chains=sa_chains,
        sa_sweeps=sa_sweeps,
        sa_beta_hot=sa_beta_hot,
        sa_beta_cold=sa_beta_cold,
        sa_schedule_type=sa_schedule_type,
        sa_custom_beta_schedule=sa_custom_beta_schedule,
        sa_num_sweeps_per_beta=sa_num_sweeps_per_beta,
        sa_randomize_order=sa_randomize_order,
        sa_acceptance_criteria=sa_acceptance_criteria,
        sa_init=sa_init,
        sa_init_md_frame=sa_init_md_frame,
        sa_restart=sa_restart,
        sa_restart_topk=sa_restart_topk,
        sa_md_sample_id=sa_md_sample_id,
        sa_md_state_ids=sa_md_state_ids,
        penalty_safety=penalty_safety,
        repair=repair,
        progress_callback=report,
    )
    sample_path = Path(str(out["sample_path"]))
    return SamplingResult(
        sample_path=sample_path,
        n_samples=int(out["n_samples"]),
        n_residues=int(out["n_residues"]),
        sa_diagnostics=out.get("sa_diagnostics"),
    )
