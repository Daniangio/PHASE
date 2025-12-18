from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PottsModel:
    """
    Potts energy:
      E(x) = sum_r h_r[x_r] + sum_(r<s in edges) J_rs[x_r, x_s]

    h: list of arrays, h[r] shape (K_r,)
    J: dict (r,s)->matrix shape (K_r,K_s), with r<s
    edges: list of (r,s) with r<s
    """
    h: List[np.ndarray]
    J: Dict[Tuple[int, int], np.ndarray]
    edges: List[Tuple[int, int]]

    def K_list(self) -> List[int]:
        return [int(v.shape[0]) for v in self.h]

    def energy(self, x: np.ndarray) -> float:
        e = 0.0
        for r, hr in enumerate(self.h):
            e += float(hr[int(x[r])])
        for (r, s) in self.edges:
            mat = self.J[(r, s)]
            e += float(mat[int(x[r]), int(x[s])])
        return e

    def energy_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (S,N)
        S, N = X.shape
        e = np.zeros(S, dtype=float)
        for r in range(N):
            e += self.h[r][X[:, r]]
        for (r, s) in self.edges:
            mat = self.J[(r, s)]
            e += mat[X[:, r], X[:, s]]
        return e

    def neighbors(self) -> List[List[int]]:
        N = len(self.h)
        neigh = [[] for _ in range(N)]
        for r, s in self.edges:
            neigh[r].append(s)
            neigh[s].append(r)
        return neigh

    def coupling(self, r: int, s: int) -> np.ndarray:
        """Return J_{rs} as a matrix with axes matching (state_r, state_s)."""
        if r == s:
            raise ValueError("No self coupling.")
        if r < s:
            return self.J[(r, s)]
        else:
            # stored as (s,r), so transpose
            return self.J[(s, r)].T


def fit_potts_pmi(
    labels: np.ndarray,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    *,
    eps_prob: float = 1e-8,
    center: bool = True,
) -> PottsModel:
    """
    Fast PMI-based initializer:
      h_r(k) = -log p_r(k)
      J_rs(k,l) = -log( p_rs(k,l) / (p_r(k)p_s(l)) )

    NOTE: This is an initializer / heuristic, not a guaranteed Potts fit.
    """
    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    K = list(map(int, K))
    edges = sorted((min(r, s), max(r, s)) for r, s in edges if r != s)

    # single-site
    p_r = []
    h = []
    for r in range(N):
        counts = np.bincount(labels[:, r], minlength=K[r]).astype(float)
        pr = (counts + eps_prob)
        pr = pr / pr.sum()
        p_r.append(pr)
        hr = -np.log(pr)
        if center:
            hr = hr - hr.mean()
        h.append(hr)

    # pairwise
    J = {}
    for r, s in edges:
        Kr, Ks = K[r], K[s]
        # fast counting
        counts = np.zeros((Kr, Ks), dtype=float)
        np.add.at(counts, (labels[:, r], labels[:, s]), 1.0)
        p_rs = (counts + eps_prob)
        p_rs = p_rs / p_rs.sum()

        p_ind = p_r[r][:, None] * p_r[s][None, :]
        ratio = p_rs / (p_ind + eps_prob)
        Jrs = -np.log(ratio + eps_prob)
        if center:
            Jrs = Jrs - Jrs.mean()
        J[(r, s)] = Jrs

    return PottsModel(h=h, J=J, edges=list(edges))


def fit_potts_pseudolikelihood_torch(
    labels: np.ndarray,
    K: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    *,
    l2: float = 1e-3,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 512,
    seed: int = 0,
    verbose: bool = True,
    init_from_pmi: bool = True,
) -> PottsModel:
    """
    True symmetric Potts fit by minimizing negative pseudolikelihood:
      sum_r -log P(x_r | x_-r; h,J)
    using PyTorch autodiff.

    This aligns with the project plan baseline (plmDCA-style), without
    the asymmetric "per-residue logistic regression then symmetrize" hassle.

    For small proteins / ~1e3 frames, this is usually fine.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for pseudolikelihood fitting. Install torch or use fit_potts_pmi.") from e

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    K = list(map(int, K))
    edges = sorted((min(r, s), max(r, s)) for r, s in edges if r != s)

    # adjacency lists by residue
    neigh = [[] for _ in range(N)]
    for r, s in edges:
        neigh[r].append(s)
        neigh[s].append(r)

    # Optional initialization from PMI heuristic
    pmi_model: PottsModel | None = None
    if init_from_pmi:
        pmi_model = fit_potts_pmi(labels, K, edges, center=True)

    # Parameters: h_r and J_rs
    torch.manual_seed(seed)
    h_params = torch.nn.ParameterList([
        torch.nn.Parameter(torch.tensor(
            -pmi_model.h[r] if pmi_model is not None else np.zeros(K[r]),
            dtype=torch.float32
        )) for r in range(N)
    ])
    J_params = torch.nn.ParameterDict()
    for r, s in edges:
        key = f"{r}_{s}"
        init_val = np.zeros((K[r], K[s])) if pmi_model is None else -pmi_model.coupling(r, s)
        J_params[key] = torch.nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    X = torch.tensor(labels, dtype=torch.long)  # (T,N)

    opt = torch.optim.Adam(list(h_params) + list(J_params.values()), lr=lr, weight_decay=l2)

    def _logits_for_residue(x_batch: "torch.Tensor", r: int) -> "torch.Tensor":
        # returns logits shape (B, K_r)
        B = x_batch.shape[0]
        logits = h_params[r].unsqueeze(0).expand(B, -1)  # (B, K_r)

        for s in neigh[r]:
            rr, ss = (r, s) if r < s else (s, r)
            key = f"{rr}_{ss}"
            Jmat = J_params[key]
            xs = x_batch[:, s]  # (B,)

            if r < s:
                # add J_rs[:, x_s]
                logits = logits + Jmat[:, xs].T
            else:
                # stored J_sr with shape (K_s,K_r)?? No: we stored (rr,ss)=(s,r) so Jmat is (K_s,K_r).
                # Need contribution as J_sr[x_s, :] which is (B,K_r)
                logits = logits + Jmat[xs, :]
        return logits

    loss_fn = torch.nn.CrossEntropyLoss()

    idx = np.arange(T)
    for ep in range(1, epochs + 1):
        rng.shuffle(idx)
        total = 0.0
        nobs = 0

        for start in range(0, T, batch_size):
            bidx = idx[start:start + batch_size]
            xb = X[bidx]  # (B,N)
            opt.zero_grad()

            loss = 0.0
            # sum over residues (pseudolikelihood)
            for r in range(N):
                logits = _logits_for_residue(xb, r)  # (B,K_r)
                y = xb[:, r]
                loss = loss + loss_fn(logits, y)

            loss.backward()
            opt.step()

            total += float(loss.item()) * len(bidx)
            nobs += len(bidx)

        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"[plm] epoch {ep:4d}/{epochs}  avg_loss={total / max(1,nobs):.6f}")

    # Export to numpy PottsModel (store couplings consistently as (r<s))
    h = [-hp.detach().cpu().numpy().astype(float) for hp in h_params]
    J = {}
    for r, s in edges:
        key = f"{r}_{s}"
        J[(r, s)] = -J_params[key].detach().cpu().numpy().astype(float)

    return PottsModel(h=h, J=J, edges=list(edges))
