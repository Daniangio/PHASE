from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from alloskin.simulation.potts_model import PottsModel
from alloskin.simulation.qubo import QUBO


def _progress_iterator(total: int, desc: str, enabled: bool) -> Iterable[int]:
    """
    Wrap range with a tqdm-style progress bar when available.
    Falls back to coarse percentage logging to avoid extra dependencies.
    """
    if not enabled:
        return range(total)

    try:
        from tqdm import trange

        return trange(total, desc=desc)
    except Exception:
        def generator():
            last_pct = -1
            for i in range(total):
                pct = int((i + 1) * 100 / max(1, total))
                if pct % 10 == 0 and pct != last_pct:
                    print(f"[{desc}] {pct}% ({i + 1}/{total})")
                    last_pct = pct
                yield i
            print(f"[{desc}] done")
        return generator()


def gibbs_sample_potts(
    model: PottsModel,
    *,
    beta: float = 1.0,
    n_samples: int = 500,
    burn_in: int = 500,
    thinning: int = 1,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    progress: bool = False,
) -> np.ndarray:
    """
    Single-site Gibbs sampler for Potts model.
    Returns samples shape (n_samples, N).
    If progress is True, shows a tqdm bar (or coarse % logs if tqdm is unavailable).
    """
    rng = np.random.default_rng(seed)
    N = len(model.h)
    K_list = model.K_list()
    neigh = model.neighbors()

    if x0 is None:
        x = np.array([rng.integers(0, K_list[r]) for r in range(N)], dtype=int)
    else:
        x = np.array(x0, dtype=int).copy()

    def conditional_probs(r: int) -> np.ndarray:
        # logits for states of residue r
        logits = -beta * model.h[r].copy()
        for s in neigh[r]:
            Jrs = model.coupling(r, s)  # (K_r, K_s)
            logits += -beta * Jrs[:, x[s]]
        # stabilize
        m = logits.max()
        p = np.exp(logits - m)
        p = p / p.sum()
        return p

    total_steps = burn_in + n_samples * thinning
    out = np.zeros((n_samples, N), dtype=int)
    oi = 0

    step_iter = _progress_iterator(total_steps, "Gibbs sweeps", progress)
    for step in step_iter:
        # sweep
        for r in range(N):
            p = conditional_probs(r)
            x[r] = int(rng.choice(len(p), p=p))
        if step >= burn_in and ((step - burn_in) % thinning == 0) and oi < n_samples:
            out[oi] = x
            oi += 1
    if hasattr(step_iter, "close") and callable(step_iter.close):
        step_iter.close()
    return out


def _qubo_adjacency(Q: Dict[Tuple[int, int], float], M: int) -> List[List[Tuple[int, float]]]:
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(M)]
    for (i, j), w in Q.items():
        adj[i].append((j, w))
        adj[j].append((i, w))
    return adj


def sa_sample_qubo_numpy(
    qubo: QUBO,
    *,
    n_reads: int = 200,
    sweeps: int = 2000,
    t_start: float = 10.0,
    t_end: float = 0.1,
    seed: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """
    Simple simulated annealing on binary QUBO.
    Returns bitstrings shape (n_reads, M).

    This is a QA-proxy sampler, not guaranteed Boltzmann sampling.
    If progress is True, shows a tqdm bar over reads and (for small n_reads) sweeps.
    """
    rng = np.random.default_rng(seed)
    M = qubo.num_vars()
    adj = _qubo_adjacency(qubo.Q, M)
    a = qubo.a

    def delta_flip(z: np.ndarray, i: int) -> float:
        # E = const + sum a_i z_i + sum_{i<j} Q_ij z_i z_j
        # flip z_i -> 1 - z_i
        zi = z[i]
        s = a[i]
        for j, w in adj[i]:
            s += w * z[j]
        # new-old multiplier
        return (1 - 2 * zi) * s

    out = np.zeros((n_reads, M), dtype=int)

    read_iter = _progress_iterator(n_reads, "SA-QUBO reads", progress)
    for r in read_iter:
        z = rng.integers(0, 2, size=M, dtype=int)

        sweep_iter = _progress_iterator(sweeps, f"SA sweeps (read {r+1}/{n_reads})", progress and n_reads <= 3)
        for sweep in sweep_iter:
            # exponential temperature schedule
            frac = sweep / max(1, sweeps - 1)
            T = t_start * (t_end / t_start) ** frac

            # random order
            for i in rng.permutation(M):
                dE = delta_flip(z, i)
                if dE <= 0:
                    z[i] = 1 - z[i]
                else:
                    if rng.random() < np.exp(-dE / max(1e-12, T)):
                        z[i] = 1 - z[i]

        out[r] = z
        if hasattr(sweep_iter, "close") and callable(sweep_iter.close):
            sweep_iter.close()
    if hasattr(read_iter, "close") and callable(read_iter.close):
        read_iter.close()
    return out


def sa_sample_qubo_neal(
    qubo: QUBO,
    *,
    n_reads: int = 200,
    sweeps: int = 2000,
    t_start: float | None = None,
    t_end: float | None = None,
    seed: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """
    Optional: use 'neal' if installed. Returns (n_reads, M) bitstrings.

    Notes:
      - neal internally chooses a schedule; t_start/t_end are accepted for API
        compatibility but not used.
      - Set progress=True to emit a short status message; neal itself does not
        stream progress per read.
    """
    try:
        import neal
        import dimod
    except Exception as e:
        raise RuntimeError("neal is not installed. pip install neal  or use sa_sample_qubo_numpy.") from e

    # Build a dimod BinaryQuadraticModel from our QUBO representation.
    linear = {i: float(qubo.a[i]) for i in range(qubo.num_vars())}
    quadratic = {(i, j): float(v) for (i, j), v in qubo.Q.items()}
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, float(qubo.const), dimod.BINARY)

    if progress:
        print(f"[neal] sampling QUBO: reads={n_reads}, sweeps={sweeps}, vars={qubo.num_vars()}")

    sampler = neal.SimulatedAnnealingSampler()
    ss = sampler.sample(
        bqm,
        num_reads=n_reads,
        num_sweeps=sweeps,
        seed=seed,
    )
    # decode samples
    arr = np.zeros((n_reads, qubo.num_vars()), dtype=int)
    for idx, sample in enumerate(ss.samples()):
        for i in range(qubo.num_vars()):
            arr[idx, i] = int(sample[i])
    return arr
