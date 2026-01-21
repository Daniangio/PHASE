from __future__ import annotations

from dataclasses import dataclass
import json
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


def save_potts_model(
    model: PottsModel,
    path: str | "os.PathLike[str]",
    *,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save a PottsModel to NPZ without pickled objects.
    """
    payload: Dict[str, np.ndarray] = {}
    n_res = len(model.h)
    payload["n_residues"] = np.array([n_res], dtype=int)
    payload["edges"] = np.asarray(model.edges, dtype=int)
    payload["K_list"] = np.asarray(model.K_list(), dtype=int)

    for idx, h in enumerate(model.h):
        payload[f"h_{idx}"] = np.asarray(h, dtype=float)
    for (r, s), mat in model.J.items():
        payload[f"J_{r}_{s}"] = np.asarray(mat, dtype=float)

    if metadata:
        payload["metadata_json"] = np.array([json.dumps(metadata)], dtype=str)

    np.savez_compressed(path, **payload)


def load_potts_model(path: str | "os.PathLike[str]") -> PottsModel:
    """
    Load a PottsModel from NPZ saved by save_potts_model.
    """
    npz = np.load(path, allow_pickle=False)
    n_res = int(npz["n_residues"][0]) if "n_residues" in npz else None
    edges = [tuple(map(int, e)) for e in npz["edges"]] if "edges" in npz else []
    if n_res is None:
        n_res = len({int(k.split("_", 1)[1]) for k in npz.files if k.startswith("h_")})

    h = [npz[f"h_{i}"] for i in range(n_res)]
    J = {}
    for r, s in edges:
        key = f"J_{r}_{s}"
        if key not in npz.files:
            raise ValueError(f"Missing coupling {key} in model NPZ.")
        J[(int(r), int(s))] = npz[key]

    return PottsModel(h=h, J=J, edges=[(int(r), int(s)) for r, s in edges])


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
    l2: float = 1e-5,
    lr: float = 1e-3,
    lr_min: float = 1e-3,
    lr_schedule: str = "cosine",
    epochs: int = 200,
    batch_size: int = 512,
    seed: int = 0,
    verbose: bool = True,
    init_from_pmi: bool = True,
    init_model: "PottsModel | None" = None,
    report_init_loss: bool = True,
    progress_callback: "callable | None" = None,
    progress_every: int = 10,
    batch_progress_callback: "callable | None" = None,
    device: str | None = None,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if verbose:
        print(f"[plm] using device={torch_device}")

    # adjacency lists by residue
    neigh = [[] for _ in range(N)]
    for r, s in edges:
        neigh[r].append(s)
        neigh[s].append(r)

    # Optional initialization from PMI heuristic
    pmi_model: PottsModel | None = init_model
    if pmi_model is None and init_from_pmi:
        pmi_model = fit_potts_pmi(labels, K, edges, center=True)

    # Parameters: h_r and J_rs
    torch.manual_seed(seed)
    h_params = torch.nn.ParameterList([
        torch.nn.Parameter(torch.tensor(
            -pmi_model.h[r] if pmi_model is not None else np.zeros(K[r]),
            dtype=torch.float32,
            device=torch_device,
        )) for r in range(N)
    ])
    J_params = torch.nn.ParameterDict()
    for r, s in edges:
        key = f"{r}_{s}"
        init_val = np.zeros((K[r], K[s])) if pmi_model is None else -pmi_model.coupling(r, s)
        J_params[key] = torch.nn.Parameter(torch.tensor(init_val, dtype=torch.float32, device=torch_device))

    X = torch.tensor(labels, dtype=torch.long, device=torch_device)  # (T,N)

    opt = torch.optim.Adam(list(h_params) + list(J_params.values()), lr=lr, weight_decay=l2)
    schedule = lr_schedule.lower() if lr_schedule else "none"
    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)
    elif schedule != "none":
        raise ValueError(f"Unknown lr_schedule={lr_schedule!r}")

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
    progress_every = max(1, int(progress_every))
    total_batches = max(1, (T + batch_size - 1) // batch_size)

    def _evaluate_avg_loss() -> float:
        total = 0.0
        nobs = 0
        with torch.no_grad():
            for start in range(0, T, batch_size):
                bidx = idx[start:start + batch_size]
                xb = X[bidx]
                loss = 0.0
                for r in range(N):
                    logits = _logits_for_residue(xb, r)
                    y = xb[:, r]
                    loss = loss + loss_fn(logits, y)
                total += float(loss.item()) * len(bidx)
                nobs += len(bidx)
        return total / max(1, nobs)

    if verbose and report_init_loss:
        init_loss = _evaluate_avg_loss()
        label = "PMI init" if pmi_model is not None else "init"
        print(f"[plm] {label} avg_loss={init_loss:.6f}")

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
            if batch_progress_callback:
                batch_num = start // batch_size + 1
                batch_progress_callback(ep, epochs, batch_num, total_batches)

        avg_loss = total / max(1, nobs)
        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"[plm] epoch {ep:4d}/{epochs}  avg_loss={avg_loss:.6f}")
        if progress_callback and (ep == 1 or ep == epochs or ep % progress_every == 0):
            progress_callback(ep, epochs, float(avg_loss))
        if scheduler is not None:
            scheduler.step()

    # Export to numpy PottsModel (store couplings consistently as (r<s))
    h = [-hp.detach().cpu().numpy().astype(float) for hp in h_params]
    J = {}
    for r, s in edges:
        key = f"{r}_{s}"
        J[(r, s)] = -J_params[key].detach().cpu().numpy().astype(float)

    return PottsModel(h=h, J=J, edges=list(edges))


def compute_pseudolikelihood_loss_torch(
    model: PottsModel,
    labels: np.ndarray,
    *,
    batch_size: int = 512,
    device: str | None = None,
) -> float:
    """
    Compute average pseudolikelihood loss for a fixed Potts model.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for pseudolikelihood loss evaluation.") from e

    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    if N != len(model.h):
        raise ValueError("Labels shape does not match Potts model size.")
    edges = sorted((min(r, s), max(r, s)) for r, s in model.edges if r != s)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    neigh = [[] for _ in range(N)]
    for r, s in edges:
        neigh[r].append(s)
        neigh[s].append(r)

    h_params = [
        torch.tensor(-model.h[r], dtype=torch.float32, device=torch_device)
        for r in range(N)
    ]
    J_params = {}
    for r, s in edges:
        key = f"{r}_{s}"
        J_params[key] = torch.tensor(-model.J[(r, s)], dtype=torch.float32, device=torch_device)

    X = torch.tensor(labels, dtype=torch.long, device=torch_device)
    loss_fn = torch.nn.CrossEntropyLoss()

    def _logits_for_residue(x_batch: "torch.Tensor", r: int) -> "torch.Tensor":
        B = x_batch.shape[0]
        logits = h_params[r].unsqueeze(0).expand(B, -1)
        for s in neigh[r]:
            rr, ss = (r, s) if r < s else (s, r)
            key = f"{rr}_{ss}"
            Jmat = J_params[key]
            xs = x_batch[:, s]
            if r < s:
                logits = logits + Jmat[:, xs].T
            else:
                logits = logits + Jmat[xs, :]
        return logits

    total = 0.0
    nobs = 0
    with torch.no_grad():
        for start in range(0, T, batch_size):
            xb = X[start:start + batch_size]
            loss = 0.0
            for r in range(N):
                logits = _logits_for_residue(xb, r)
                y = xb[:, r]
                loss = loss + loss_fn(logits, y)
            total += float(loss.item()) * xb.shape[0]
            nobs += xb.shape[0]

    return total / max(1, nobs)
