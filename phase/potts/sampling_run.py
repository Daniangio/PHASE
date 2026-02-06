from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from phase.io.data import load_npz
from phase.potts.potts_model import PottsModel, add_potts_models, load_potts_model
from phase.potts.qubo import decode_onehot, encode_onehot, potts_to_qubo_onehot
from phase.potts.sample_io import SAMPLE_NPZ_FILENAME, save_sample_npz
from phase.potts.sampling import (
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
) -> np.ndarray:
    mode = (mode or "md").lower()
    if mode in {"md", "md-frame"}:
        if md_labels is None or md_labels.size == 0:
            if mode == "md-frame":
                raise ValueError("SA init set to md-frame, but MD labels are unavailable.")
            return _sample_labels_from_fields(model, beta=beta, n_samples=n_reads, rng=rng)
        if mode == "md-frame":
            if md_frame < 0:
                raise ValueError("--sa-init md-frame requires --sa-init-md-frame >= 0.")
            if md_frame >= md_labels.shape[0]:
                raise ValueError(f"--sa-init-md-frame {md_frame} out of range (0..{md_labels.shape[0]-1}).")
            return np.repeat(md_labels[md_frame : md_frame + 1], n_reads, axis=0)
        idx = rng.integers(0, md_labels.shape[0], size=n_reads)
        return md_labels[idx]
    if mode in {"random-h", "h"}:
        return _sample_labels_from_fields(model, beta=beta, n_samples=n_reads, rng=rng)
    if mode in {"random-uniform", "uniform"}:
        return _sample_labels_uniform(model.K_list(), n_reads, rng)
    raise ValueError(f"Unknown sa-init mode: {mode}")


def _parse_float_list(raw: str) -> List[float]:
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    return [float(p) for p in parts]


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

def _filter_md_labels_for_states(
    labels: np.ndarray,
    frame_state_ids: np.ndarray | None,
    *,
    state_ids: Sequence[str] | None,
) -> np.ndarray:
    if labels is None:
        return labels
    if not state_ids:
        return labels
    if frame_state_ids is None:
        raise ValueError("MD frame_state_ids missing in cluster NPZ; cannot filter by --sa-md-state-ids.")
    ids = [str(s).strip() for s in state_ids if str(s).strip()]
    if not ids:
        return labels
    frame_ids = np.asarray(frame_state_ids).astype(str)
    mask = np.isin(frame_ids, ids)
    if not np.any(mask):
        raise ValueError(f"No MD frames matched sa_md_state_ids={ids}.")
    return np.asarray(labels)[mask]


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


def _run_sa_independent_worker(payload: dict[str, object]) -> dict[str, object]:
    model = _load_combined_model(payload["model_npz"])  # type: ignore[arg-type]
    beta = float(payload["beta"])
    penalty_safety = float(payload["penalty_safety"])
    n_reads = int(payload["n_reads"])
    sweeps = int(payload["sweeps"])
    seed = int(payload["seed"])
    beta_range = payload.get("beta_range")  # type: ignore[assignment]
    sa_init = str(payload.get("sa_init", "md"))
    sa_init_md_frame = int(payload.get("sa_init_md_frame", -1))
    repair = str(payload.get("repair", "none"))

    md_ds = load_npz(str(payload["cluster_npz"]), unassigned_policy="drop_frames", allow_missing_edges=True)  # type: ignore[arg-type]
    md_labels = md_ds.labels
    md_state_ids = _parse_str_list(str(payload.get("sa_md_state_ids", "")))
    md_labels = _filter_md_labels_for_states(md_labels, md_ds.frame_state_ids, state_ids=md_state_ids)

    qubo = potts_to_qubo_onehot(model, beta=beta, penalty_safety=penalty_safety)
    init_rng = np.random.default_rng(seed + 1000)
    init_labels = _build_sa_initial_labels(
        mode=sa_init,
        md_labels=md_labels,
        model=model,
        beta=beta,
        n_reads=n_reads,
        md_frame=sa_init_md_frame,
        rng=init_rng,
    )
    init_states = encode_onehot(init_labels, qubo) if init_labels is not None and init_labels.size else None

    Z = sa_sample_qubo_neal(
        qubo,
        n_reads=n_reads,
        sweeps=sweeps,
        seed=seed,
        progress=False,
        beta_range=beta_range,  # type: ignore[arg-type]
        initial_states=init_states,
    )
    labels, invalid_mask, valid_counts = _sa_decode_labels(Z, qubo, repair=repair)
    return {"labels": labels, "invalid_mask": invalid_mask, "valid_counts": valid_counts}


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
    sa_init = str(payload.get("sa_init", "md"))
    sa_init_md_frame = int(payload.get("sa_init_md_frame", -1))
    sa_restart = str(payload.get("sa_restart", "previous")).strip().lower()
    repair = str(payload.get("repair", "none"))

    if sa_restart not in {"previous", "md"}:
        raise ValueError("--sa-restart must be one of: previous, md (for chain mode).")

    md_ds = load_npz(str(payload["cluster_npz"]), unassigned_policy="drop_frames", allow_missing_edges=True)  # type: ignore[arg-type]
    md_labels = md_ds.labels
    md_state_ids = _parse_str_list(str(payload.get("sa_md_state_ids", "")))
    md_labels = _filter_md_labels_for_states(md_labels, md_ds.frame_state_ids, state_ids=md_state_ids)

    # Build QUBO and corresponding BQM once.
    qubo = potts_to_qubo_onehot(model, beta=beta, penalty_safety=penalty_safety)
    linear = {i: float(qubo.a[i]) for i in range(qubo.num_vars())}
    quadratic = {k: float(v) for k, v in qubo.Q.items()}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, float(qubo.const), dimod.BINARY)
    sampler = neal.SimulatedAnnealingSampler()

    init_rng = np.random.default_rng(seed + 1000)
    next_init = _build_sa_initial_labels(
        mode=sa_init,
        md_labels=md_labels,
        model=model,
        beta=beta,
        n_reads=1,
        md_frame=sa_init_md_frame,
        rng=init_rng,
    )[0]

    repair_mode = None if str(repair) == "none" else str(repair)
    labels = np.zeros((n_samples, len(qubo.var_slices)), dtype=np.int32)
    valid_counts = np.zeros(n_samples, dtype=np.int32)
    invalid_mask = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
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
            "num_sweeps": sweeps,
            "seed": int(seed) + int(i),
            "initial_states": init,
        }
        if beta_range is not None:
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
            next_init = x
        else:
            # fresh MD init for each sample
            next_init = _build_sa_initial_labels(
                mode="md",
                md_labels=md_labels,
                model=model,
                beta=beta,
                n_reads=1,
                md_frame=-1,
                rng=init_rng,
            )[0]

    return {"labels": labels, "invalid_mask": invalid_mask, "valid_counts": valid_counts}


def run_sampling(
    *,
    cluster_npz: str,
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
    sa_beta_hot: float = 0.0,
    sa_beta_cold: float = 0.0,
    sa_init: str = "md",
    sa_init_md_frame: int = -1,
    sa_restart: str = "previous",
    sa_restart_topk: int = 200,
    sa_md_state_ids: str = "",
    penalty_safety: float = 3.0,
    repair: str = "none",
    progress_callback: Callable[[str, int], None] | None = None,
) -> SamplingResult:
    """
    Run a sampler and write results_dir/sample.npz (minimal sample schema).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_path = results_dir / SAMPLE_NPZ_FILENAME

    def report(msg: str, pct: int) -> None:
        if progress_callback:
            progress_callback(msg, int(pct))

    model = _load_combined_model(model_npz)

    method = (sampling_method or "gibbs").strip().lower()
    if method not in {"gibbs", "sa"}:
        raise ValueError("--sampling-method must be gibbs or sa.")

    if method == "gibbs":
        gm = (gibbs_method or "single").strip().lower()
        if gm not in {"single", "rex"}:
            raise ValueError("--gibbs-method must be single or rex.")

        if gm == "single":
            # Optionally split across independent chains and concatenate.
            n_chains = max(1, int(gibbs_chains))
            total_samples = max(0, int(gibbs_samples))
            if n_chains > max(1, total_samples):
                n_chains = max(1, total_samples)
            base = total_samples // n_chains if n_chains else total_samples
            extra = total_samples % n_chains if n_chains else 0
            chain_samples = [base + (1 if i < extra else 0) for i in range(n_chains)]
            parts: List[np.ndarray] = []
            for idx, n_samp in enumerate(chain_samples):
                if n_samp <= 0:
                    continue
                report(f"Gibbs sampling chain {idx + 1}/{n_chains}", 10 + int(80 * idx / max(1, n_chains)))
                part = gibbs_sample_potts(
                    model,
                    beta=float(beta),
                    n_samples=int(n_samp),
                    burn_in=int(gibbs_burnin),
                    thinning=int(gibbs_thin),
                    seed=int(seed) + idx,
                    progress=bool(progress),
                    progress_mode="samples",
                    progress_desc=f"Gibbs chain {idx + 1}/{n_chains} samples",
                    progress_position=idx if progress and n_chains > 1 else None,
                )
                if part.size:
                    parts.append(part)
            labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(model.h)), dtype=np.int32)
            save_sample_npz(sample_path, labels=labels)
            return SamplingResult(sample_path=sample_path, n_samples=int(labels.shape[0]), n_residues=int(labels.shape[1]))

        # Replica exchange
        if rex_betas.strip():
            betas = _parse_float_list(rex_betas)
        else:
            betas = make_beta_ladder(
                beta_min=float(rex_beta_min),
                beta_max=float(rex_beta_max),
                n_replicas=int(rex_n_replicas),
                spacing=str(rex_spacing),
            )
        if all(abs(b - float(beta)) > 1e-12 for b in betas):
            betas = sorted(set(betas + [float(beta)]))
        betas = [float(b) for b in betas]

        total_rounds = max(1, int(rex_rounds))
        n_chains = max(1, int(rex_chains))
        if n_chains > total_rounds:
            n_chains = total_rounds
        if n_chains > 1:
            base_rounds = total_rounds // n_chains
            extra = total_rounds % n_chains
            chain_rounds = [base_rounds + (1 if i < extra else 0) for i in range(n_chains)]
        else:
            chain_rounds = [total_rounds]

        chain_runs: List[dict[str, object] | None] = [None] * n_chains
        burnin_clipped = False

        if n_chains == 1:
            burn_in = min(int(rex_burnin_rounds), max(0, total_rounds - 1))
            burnin_clipped = burn_in != int(rex_burnin_rounds)
            chain_runs[0] = replica_exchange_gibbs_potts(
                model,
                betas=betas,
                sweeps_per_round=int(rex_sweeps_per_round),
                n_rounds=int(total_rounds),
                burn_in_rounds=int(burn_in),
                thinning_rounds=int(rex_thin_rounds),
                seed=int(seed),
                progress=bool(progress),
                progress_mode="samples",
                progress_desc="REX samples",
            )
        else:
            with ProcessPoolExecutor(max_workers=n_chains) as executor:
                futures = {}
                for idx in range(n_chains):
                    rounds = int(chain_rounds[idx])
                    burn_in = min(int(rex_burnin_rounds), max(0, rounds - 1))
                    futures[executor.submit(
                        _run_rex_chain_worker,
                        {
                            "model": model,
                            "betas": betas,
                            "sweeps_per_round": int(rex_sweeps_per_round),
                            "n_rounds": rounds,
                            "burn_in_rounds": burn_in,
                            "thinning_rounds": int(rex_thin_rounds),
                            "seed": int(seed) + idx,
                            "progress": bool(progress),
                            "progress_every": max(1, rounds // 20) if rounds else 1,
                            "progress_mode": "samples",
                            "progress_desc": f"REX chain {idx + 1}/{n_chains} samples",
                            "progress_position": idx,
                        },
                    )] = (idx, burn_in != int(rex_burnin_rounds))

                completed = 0
                for future in as_completed(futures):
                    idx, clipped = futures[future]
                    chain_runs[idx] = future.result()
                    burnin_clipped = burnin_clipped or clipped
                    completed += 1
                    report(f"Replica exchange chains {completed}/{n_chains}", 10 + int(80 * completed / max(1, n_chains)))

        if burnin_clipped:
            print("[rex] note: burn-in rounds truncated for short chains.")

        parts = []
        for run in chain_runs:
            if not isinstance(run, dict):
                continue
            samples_by_beta = run.get("samples_by_beta")
            if not isinstance(samples_by_beta, dict):
                continue
            arr = samples_by_beta.get(float(beta))
            if isinstance(arr, np.ndarray) and arr.size:
                parts.append(arr)
        labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, len(model.h)), dtype=np.int32)
        save_sample_npz(sample_path, labels=labels)
        return SamplingResult(sample_path=sample_path, n_samples=int(labels.shape[0]), n_residues=int(labels.shape[1]))

    # SA/QUBO
    if (sa_beta_hot and not sa_beta_cold) or (sa_beta_cold and not sa_beta_hot):
        raise ValueError("Provide both --sa-beta-hot and --sa-beta-cold, or neither.")
    beta_range = None
    if sa_beta_hot and sa_beta_cold:
        beta_range = (float(sa_beta_hot), float(sa_beta_cold))

    restart = str(sa_restart or "previous").strip().lower()
    if restart not in {"previous", "md", "independent"}:
        raise ValueError("--sa-restart must be one of: previous, md, independent.")

    n_chains = max(1, int(sa_chains))
    total_reads = max(0, int(sa_reads))
    if total_reads <= 0:
        labels = np.zeros((0, len(model.h)), dtype=np.int32)
        save_sample_npz(sample_path, labels=labels)
        return SamplingResult(sample_path=sample_path, n_samples=0, n_residues=int(labels.shape[1]))
    if n_chains > max(1, total_reads):
        n_chains = max(1, total_reads)
    base = total_reads // n_chains if n_chains else total_reads
    extra = total_reads % n_chains if n_chains else 0
    chain_reads = [base + (1 if i < extra else 0) for i in range(n_chains)]
    worker_fn = _run_sa_independent_worker if restart == "independent" else _run_sa_chain_worker

    def _payload(idx: int, n: int) -> dict[str, object]:
        common: dict[str, object] = {
            "model_npz": model_npz,
            "cluster_npz": cluster_npz,
            "beta": float(beta),
            "penalty_safety": float(penalty_safety),
            "sweeps": int(sa_sweeps),
            "seed": int(seed) + idx,
            "beta_range": beta_range,
            "sa_init": str(sa_init),
            "sa_init_md_frame": int(sa_init_md_frame),
            "sa_md_state_ids": str(sa_md_state_ids),
            "repair": str(repair),
        }
        if restart == "independent":
            common["n_reads"] = int(n)
        else:
            common["n_samples"] = int(n)
            common["sa_restart"] = restart
        return common

    parts_labels: List[np.ndarray] = []
    parts_invalid: List[np.ndarray] = []
    parts_valid_counts: List[np.ndarray] = []

    if n_chains <= 1:
        out = worker_fn(_payload(0, int(total_reads)))
        parts_labels.append(np.asarray(out["labels"], dtype=np.int32))
        parts_invalid.append(np.asarray(out["invalid_mask"], dtype=bool))
        parts_valid_counts.append(np.asarray(out["valid_counts"], dtype=np.int32))
    else:
        with ProcessPoolExecutor(max_workers=n_chains) as executor:
            futures = {}
            for idx, n in enumerate(chain_reads):
                if int(n) <= 0:
                    continue
                futures[executor.submit(worker_fn, _payload(idx, int(n)))] = idx
            completed = 0
            for future in as_completed(futures):
                out = future.result()
                parts_labels.append(np.asarray(out["labels"], dtype=np.int32))
                parts_invalid.append(np.asarray(out["invalid_mask"], dtype=bool))
                parts_valid_counts.append(np.asarray(out["valid_counts"], dtype=np.int32))
                completed += 1
                report(f"SA chains {completed}/{n_chains}", 10 + int(80 * completed / max(1, n_chains)))

    labels = np.concatenate(parts_labels, axis=0) if parts_labels else np.zeros((0, len(model.h)), dtype=np.int32)
    invalid_mask = np.concatenate(parts_invalid, axis=0) if parts_invalid else np.zeros((labels.shape[0],), dtype=bool)
    valid_counts = (
        np.concatenate(parts_valid_counts, axis=0) if parts_valid_counts else np.zeros((labels.shape[0],), dtype=np.int32)
    )

    save_sample_npz(sample_path, labels=labels, invalid_mask=invalid_mask, valid_counts=valid_counts)
    return SamplingResult(sample_path=sample_path, n_samples=int(labels.shape[0]), n_residues=int(labels.shape[1]))
