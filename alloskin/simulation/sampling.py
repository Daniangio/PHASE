from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from alloskin.simulation.potts_model import PottsModel
from alloskin.simulation.qubo import QUBO


def _progress_iterator(total: int, desc: str, enabled: bool) -> Iterable[int]:
    """
    Wrap range with a tqdm-style progress bar when available.
    Falls back to a plain range if tqdm is unavailable.
    """
    if not enabled:
        return range(total)
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(range(total), total=total, desc=desc)
    except Exception:
        # No tqdm; just return a normal iterator.
        return range(total)


def _gibbs_one_sweep(
    model: PottsModel,
    x: np.ndarray,
    *,
    beta: float,
    rng: np.random.Generator,
) -> None:
    """
    In-place one full Gibbs sweep over all residues (single-site updates).
    """
    N = len(model.h)
    neigh = model.neighbors()

    for r in range(N):
        # logits for states of residue r
        logits = -beta * model.h[r].copy()
        for s in neigh[r]:
            Jrs = model.coupling(r, s)  # (K_r, K_s)
            logits += -beta * Jrs[:, x[s]]

        # stabilize and sample
        m = logits.max()
        p = np.exp(logits - m)
        p = p / p.sum()
        x[r] = int(rng.choice(len(p), p=p))


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
    """
    rng = np.random.default_rng(seed)
    N = len(model.h)
    K_list = model.K_list()

    if x0 is None:
        x = np.array([rng.integers(0, K_list[r]) for r in range(N)], dtype=int)
    else:
        x = np.array(x0, dtype=int).copy()

    total_steps = burn_in + n_samples * thinning
    out = np.zeros((n_samples, N), dtype=int)
    oi = 0

    step_iter = _progress_iterator(total_steps, "Gibbs sweeps", progress)
    for step in step_iter:
        _gibbs_one_sweep(model, x, beta=beta, rng=rng)

        if step >= burn_in and ((step - burn_in) % thinning == 0) and oi < n_samples:
            out[oi] = x
            oi += 1

        if oi >= n_samples:
            break

    return out


def make_beta_ladder(
    *,
    beta_min: float,
    beta_max: float,
    n_replicas: int,
    spacing: str = "geom",
) -> List[float]:
    """
    Construct a monotone beta ladder for replica exchange.
    spacing:
      - "geom": geometric spacing (good default)
      - "lin": linear spacing
    """
    if n_replicas < 2:
        raise ValueError("n_replicas must be >= 2")
    if beta_min <= 0 or beta_max <= 0 or beta_min >= beta_max:
        raise ValueError("Require 0 < beta_min < beta_max")

    if spacing == "geom":
        betas = np.geomspace(beta_min, beta_max, n_replicas)
    elif spacing == "lin":
        betas = np.linspace(beta_min, beta_max, n_replicas)
    else:
        raise ValueError(f"Unknown spacing={spacing!r}")

    return [float(b) for b in betas]


def replica_exchange_gibbs_potts(
    model: PottsModel,
    *,
    betas: Sequence[float],
    sweeps_per_round: int = 2,
    n_rounds: int = 2000,
    burn_in_rounds: int = 500,
    thinning_rounds: int = 1,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    progress: bool = False,
) -> Dict[str, object]:
    """
    Parallel tempering (replica exchange) for the Potts model.

    We run replicas at different inverse temperatures (betas).
    Each round:
      1) do 'sweeps_per_round' Gibbs sweeps in each replica at its own beta,
      2) attempt swaps between adjacent betas.

    Returns a dict:
      - "betas": list[float]
      - "samples_by_beta": dict[float -> np.ndarray shape (S,N)]
      - "swap_accept_rate": np.ndarray shape (n_replicas-1,)
      - "energy_traces": dict[float -> np.ndarray shape (n_saved,)]  (energies at save times)
    """
    betas = [float(b) for b in betas]
    if sorted(betas) != list(betas):
        raise ValueError("betas must be sorted ascending.")
    if len(betas) < 2:
        raise ValueError("Need at least 2 betas for replica exchange.")
    if any(b <= 0 for b in betas):
        raise ValueError("All betas must be > 0")

    rng = np.random.default_rng(seed)
    N = len(model.h)
    K_list = model.K_list()

    # Initialize replicas (states)
    replicas: List[np.ndarray] = []
    if x0 is None:
        for _ in betas:
            replicas.append(np.array([rng.integers(0, K_list[r]) for r in range(N)], dtype=int))
    else:
        x0 = np.array(x0, dtype=int).copy()
        for _ in betas:
            replicas.append(x0.copy())

    # Current energies (raw, unscaled by beta)
    energies = np.array([model.energy(x) for x in replicas], dtype=float)

    # Storage
    samples_by_beta: Dict[float, List[np.ndarray]] = {b: [] for b in betas}
    energy_traces: Dict[float, List[float]] = {b: [] for b in betas}
    accept = np.zeros(len(betas) - 1, dtype=int)
    trials = np.zeros(len(betas) - 1, dtype=int)

    round_iter = _progress_iterator(n_rounds, "Replica-exchange rounds", progress)
    for rnd in round_iter:
        # 1) local updates
        for i, b in enumerate(betas):
            for _ in range(sweeps_per_round):
                _gibbs_one_sweep(model, replicas[i], beta=b, rng=rng)
            energies[i] = model.energy(replicas[i])

        # 2) swap attempts (adjacent)
        # We attempt swaps in alternating pattern to reduce bias:
        # even pairs on even rounds, odd pairs on odd rounds.
        start = 0 if (rnd % 2 == 0) else 1
        for i in range(start, len(betas) - 1, 2):
            b_i, b_j = betas[i], betas[i + 1]
            e_i, e_j = energies[i], energies[i + 1]

            # acceptance prob for swapping configurations between betas
            # alpha = min(1, exp((beta_i - beta_j) * (E(x_i) - E(x_j))))
            d = (b_i - b_j) * (e_i - e_j)
            trials[i] += 1
            if d >= 0 or rng.random() < np.exp(d):
                # swap states and energies
                replicas[i], replicas[i + 1] = replicas[i + 1], replicas[i]
                energies[i], energies[i + 1] = energies[i + 1], energies[i]
                accept[i] += 1

        # Save after burn-in, with thinning
        if rnd >= burn_in_rounds and ((rnd - burn_in_rounds) % thinning_rounds == 0):
            for i, b in enumerate(betas):
                samples_by_beta[b].append(replicas[i].copy())
                energy_traces[b].append(float(energies[i]))

    # Convert lists to arrays
    samples_by_beta_arr: Dict[float, np.ndarray] = {}
    energy_traces_arr: Dict[float, np.ndarray] = {}
    for b in betas:
        samples_by_beta_arr[b] = np.stack(samples_by_beta[b], axis=0) if len(samples_by_beta[b]) else np.zeros((0, N), dtype=int)
        energy_traces_arr[b] = np.array(energy_traces[b], dtype=float)

    swap_accept_rate = np.divide(accept, np.maximum(1, trials)).astype(float)

    return {
        "betas": betas,
        "samples_by_beta": samples_by_beta_arr,
        "swap_accept_rate": swap_accept_rate,
        "energy_traces": energy_traces_arr,
    }

def sa_sample_qubo_neal(
    qubo: QUBO,
    *,
    n_reads: int = 200,
    sweeps: int = 2000,
    seed: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """
    Use neal (if installed) to sample a QUBO/BQM.

    NOTE: neal's SA schedule differs from our numpy implementation; do not interpret
    its sweeps/temperature as directly comparable to a physical beta.
    """
    try:
        import dimod  # type: ignore
        import neal  # type: ignore
    except Exception as e:
        raise RuntimeError("neal/dimod are not installed.") from e

    # Convert our QUBO to a dimod BinaryQuadraticModel
    linear = {i: float(qubo.a[i]) for i in range(qubo.num_vars())}
    quadratic = {k: float(v) for k, v in qubo.Q.items()}
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

    arr = np.zeros((n_reads, qubo.num_vars()), dtype=int)
    for idx, sample in enumerate(ss.samples()):
        for i in range(qubo.num_vars()):
            arr[idx, i] = int(sample[i])
    return arr
