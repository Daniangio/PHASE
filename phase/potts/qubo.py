from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from phase.potts.potts_model import PottsModel


@dataclass(frozen=True)
class QUBO:
    """
    Energy:
      E(z) = const + sum_i a[i] z_i + sum_{i<j} Q[(i,j)] z_i z_j
    where z_i in {0,1}.
    """
    a: np.ndarray
    Q: Dict[Tuple[int, int], float]
    const: float
    # for decoding:
    var_slices: List[slice]  # per residue r -> slice in [0,M)
    K_list: List[int]

    def num_vars(self) -> int:
        return int(self.a.shape[0])

    def energy(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=int)
        e = float(self.const + np.dot(self.a, z))
        for (i, j), w in self.Q.items():
            e += w * z[i] * z[j]
        return e

    def constraint_violations(self, z: np.ndarray) -> np.ndarray:
        """
        returns per-residue (sum z_rk - 1)
        """
        z = np.asarray(z, dtype=int)
        viol = []
        for sl in self.var_slices:
            viol.append(int(z[sl].sum()) - 1)
        return np.array(viol, dtype=int)


def _build_slices(K_list: Sequence[int]) -> List[slice]:
    out = []
    start = 0
    for K in K_list:
        out.append(slice(start, start + int(K)))
        start += int(K)
    return out


def adaptive_penalties(model: PottsModel, safety: float = 3.0) -> np.ndarray:
    """
    Residue-specific lambda_r ≥ max|h_r| + sum_neighbors max|J_rs|.
    """
    N = len(model.h)
    neigh = model.neighbors()
    lam = np.zeros(N, dtype=float)
    for r in range(N):
        bound = float(np.max(np.abs(model.h[r])))
        for s in neigh[r]:
            Jrs = model.coupling(r, s)
            bound += float(np.max(np.abs(Jrs)))
        lam[r] = safety * max(1e-6, bound)
    return lam


def potts_to_qubo_onehot(
    model: PottsModel,
    *,
    beta: float = 1.0,
    penalty_lambda: np.ndarray | None = None,
    penalty_safety: float = 3.0,
) -> QUBO:
    """
    Map Potts to QUBO with one-hot constraints per residue:
      z_{r,k} ∈ {0,1}, sum_k z_{r,k} = 1.
    Penalty: λ_r (sum_k z_{r,k} - 1)^2.

    We incorporate beta by scaling energies with beta in the QUBO.
    """
    K_list = model.K_list()
    var_slices = _build_slices(K_list)
    M = sum(K_list)

    if penalty_lambda is None:
        penalty_lambda = adaptive_penalties(model, safety=penalty_safety)
    penalty_lambda = np.asarray(penalty_lambda, dtype=float)
    if penalty_lambda.shape != (len(K_list),):
        raise ValueError("penalty_lambda must have shape (N,)")

    a = np.zeros(M, dtype=float)
    Q: Dict[Tuple[int, int], float] = {}
    const = 0.0

    # linear Potts (scaled by beta)
    for r, sl in enumerate(var_slices):
        a[sl] += beta * model.h[r]

    # quadratic Potts couplings
    for (r, s) in model.edges:
        Jr = model.J[(r, s)]  # (K_r, K_s)
        sl_r = var_slices[r]
        sl_s = var_slices[s]
        for kr in range(Jr.shape[0]):
            for ks in range(Jr.shape[1]):
                i = sl_r.start + kr
                j = sl_s.start + ks
                if i < j:
                    Q[(i, j)] = Q.get((i, j), 0.0) + beta * float(Jr[kr, ks])
                else:
                    Q[(j, i)] = Q.get((j, i), 0.0) + beta * float(Jr[kr, ks])

    # one-hot penalties
    # (sum z - 1)^2 = - sum z + 2 sum_{k<k'} z_k z_k' + 1
    for r, sl in enumerate(var_slices):
        lam = float(penalty_lambda[r])

        # linear: -lam per variable
        a[sl] += -lam

        # quadratic within residue: +2*lam for each pair
        idxs = list(range(sl.start, sl.stop))
        for i_pos in range(len(idxs)):
            for j_pos in range(i_pos + 1, len(idxs)):
                i, j = idxs[i_pos], idxs[j_pos]
                Q[(i, j)] = Q.get((i, j), 0.0) + 2.0 * lam

        # constant: +lam
        const += lam

    return QUBO(a=a, Q=Q, const=const, var_slices=var_slices, K_list=K_list)


def decode_onehot(
    z: np.ndarray,
    qubo: QUBO,
    *,
    repair: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode z -> x (labels per residue).
    Returns (x, valid_mask_per_residue).
    valid if exactly one 1 in the residue slice.
    If repair == "argmax": choose the index of max(z_slice) (ties -> first).
    """
    z = np.asarray(z, dtype=int)
    x = np.zeros(len(qubo.var_slices), dtype=int)
    valid = np.zeros(len(qubo.var_slices), dtype=bool)

    for r, sl in enumerate(qubo.var_slices):
        s = z[sl]
        ones = np.where(s == 1)[0]
        if len(ones) == 1:
            x[r] = int(ones[0])
            valid[r] = True
        else:
            valid[r] = False
            if repair == "argmax":
                x[r] = int(np.argmax(s))
            else:
                # arbitrary but explicit:
                x[r] = 0
    return x, valid


def encode_onehot(
    x: np.ndarray,
    qubo: QUBO,
) -> np.ndarray:
    """
    Encode label assignments into a one-hot QUBO bitstring.
    Accepts x shaped (N,) or (S, N) where N = # residues.
    Returns z shaped (M,) or (S, M) where M = # QUBO variables.
    """
    x = np.asarray(x, dtype=int)
    single = False
    if x.ndim == 1:
        single = True
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != len(qubo.var_slices):
        raise ValueError("encode_onehot expects shape (N,) or (S, N) matching qubo.var_slices.")

    n_samples = x.shape[0]
    z = np.zeros((n_samples, qubo.num_vars()), dtype=int)
    for r, sl in enumerate(qubo.var_slices):
        idx = sl.start + x[:, r]
        z[np.arange(n_samples), idx] = 1
    return z[0] if single else z
