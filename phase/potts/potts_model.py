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


def load_potts_model_metadata(path: str | "os.PathLike[str]") -> Optional[dict]:
    """Load metadata_json from a Potts model NPZ, if present."""
    npz = np.load(path, allow_pickle=False)
    if "metadata_json" not in npz:
        return None
    raw = npz["metadata_json"]
    try:
        payload = raw.item() if hasattr(raw, "item") else raw
        return json.loads(str(payload))
    except Exception:
        return None


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
    lambda_J_block: float = 1e-4,
    zero_sum_gauge: bool = True,
    lr: float = 1e-2,
    lr_min: float = 1e-4,
    lr_schedule: str = "cosine",
    epochs: int = 200,
    batch_size: int = 512,
    seed: int = 0,
    verbose: bool = True,
    init_from_pmi: bool = True,
    init_model: "PottsModel | None" = None,
    report_init_loss: bool = True,
    val_labels: Optional[np.ndarray] = None,
    start_best_loss: Optional[float] = None,
    start_best_val_loss: Optional[float] = None,
    best_model_path: str | "os.PathLike[str]" | None = None,
    best_model_metadata: Optional[dict] = None,
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
    X_val = None
    if val_labels is not None:
        val_labels = np.asarray(val_labels, dtype=int)
        if val_labels.ndim != 2 or val_labels.shape[1] != N:
            raise ValueError("val_labels must match shape (T_val, N).")
        X_val = torch.tensor(val_labels, dtype=torch.long, device=torch_device)

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

    eps = 1e-12

    def _fro_norm(mat: "torch.Tensor") -> "torch.Tensor":
        # Frobenius norm (smooth at 0)
        return torch.sqrt(torch.sum(mat * mat) + eps)

    @torch.no_grad()
    def _apply_zero_sum_gauge_():
        """
        Put (h, J) into a standard zero-sum gauge while preserving the model distribution
        (up to additive constants). Works on the PARAMS USED IN LOGITS (h_params, J_params).

        For each edge (r,s) with matrix M (K_r x K_s):
            row_mean[a] = mean_b M[a,b]
            col_mean[b] = mean_a M[a,b]
            overall = mean_{a,b} M[a,b]
        We set:
            M <- M - row_mean - col_mean + overall
            h_r <- h_r + row_mean
            h_s <- h_s + col_mean
        Then finally center h_r to zero-mean (optional gauge fixing on fields).
        """
        # First, fix each coupling block and push row/col means into fields
        for (r, s) in edges:
            key = f"{r}_{s}"
            M = J_params[key]                    # shape (K_r, K_s)

            row_mean = M.mean(dim=1, keepdim=True)  # (K_r, 1)
            col_mean = M.mean(dim=0, keepdim=True)  # (1, K_s)
            overall  = M.mean()                      # scalar

            # push means into the corresponding fields (preserves energies up to constants)
            h_params[r].add_(row_mean.squeeze(1))
            h_params[s].add_(col_mean.squeeze(0))

            # double-center coupling
            M.sub_(row_mean)
            M.sub_(col_mean)
            M.add_(overall)

        # Then center fields to zero mean per site (pure gauge; adds constants)
        for r in range(N):
            h_params[r].sub_(h_params[r].mean())

    idx = np.arange(T)
    progress_every = max(1, int(progress_every))
    total_batches = max(1, (T + batch_size - 1) // batch_size)

    def _evaluate_avg_loss(X_eval: "torch.Tensor") -> float:
        total = 0.0
        nobs = 0
        with torch.no_grad():
            n_eval = X_eval.shape[0]
            for start in range(0, n_eval, batch_size):
                xb = X_eval[start:start + batch_size]
                loss = 0.0
                for r in range(N):
                    logits = _logits_for_residue(xb, r)
                    y = xb[:, r]
                    loss = loss + loss_fn(logits, y)
                total += float(loss.item()) * xb.shape[0]
                nobs += xb.shape[0]
        return total / max(1, nobs)

    if verbose and report_init_loss:
        init_loss = _evaluate_avg_loss(X)
        label = "PMI init" if pmi_model is not None else "init"
        print(f"[plm] {label} avg_loss={init_loss:.6f}")
        if X_val is not None:
            init_val_loss = _evaluate_avg_loss(X_val)
            print(f"[plm] {label} val_loss={init_val_loss:.6f}")

    best_metric = start_best_val_loss if X_val is not None else start_best_loss
    best_loss = start_best_loss
    best_val_loss = start_best_val_loss
    best_epoch = None
    last_loss = None
    last_val_loss = None

    def _export_model() -> PottsModel:
        h = [-hp.detach().cpu().numpy().astype(float) for hp in h_params]
        J: Dict[Tuple[int, int], np.ndarray] = {}
        for r, s in edges:
            key = f"{r}_{s}"
            J[(r, s)] = -J_params[key].detach().cpu().numpy().astype(float)
        return PottsModel(h=h, J=J, edges=list(edges))

    def _maybe_save_best(ep: int, train_loss: float, val_loss: Optional[float]) -> None:
        nonlocal best_metric, best_loss, best_val_loss, best_epoch
        metric = val_loss if X_val is not None else train_loss
        if metric is None:
            return
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_loss = train_loss
            best_val_loss = val_loss
            best_epoch = ep
            if best_model_path is not None:
                meta = dict(best_model_metadata or {})
                meta.update(
                    {
                        "plm_best_loss": best_loss,
                        "plm_best_val_loss": best_val_loss,
                        "plm_best_epoch": best_epoch,
                    }
                )
                save_potts_model(_export_model(), best_model_path, metadata=meta)

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
            
            # block regularization per edge (group-lasso style over coupling matrices)
            if lambda_J_block > 0:
                reg = torch.tensor(0.0, device=torch_device)
                for key in J_params:
                    reg = reg + _fro_norm(J_params[key])
                loss = loss + lambda_J_block * reg

            loss.backward()

            opt.step()

            if zero_sum_gauge:
                _apply_zero_sum_gauge_()

            total += float(loss.item()) * len(bidx)
            nobs += len(bidx)
            if batch_progress_callback:
                batch_num = start // batch_size + 1
                batch_progress_callback(ep, epochs, batch_num, total_batches)

        avg_loss = total / max(1, nobs)
        last_loss = float(avg_loss)
        val_loss = None
        if X_val is not None:
            val_loss = float(_evaluate_avg_loss(X_val))
            last_val_loss = val_loss
        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"[plm] epoch {ep:4d}/{epochs}  avg_loss={avg_loss:.6f}")
        if progress_callback and (ep == 1 or ep == epochs or ep % progress_every == 0):
            progress_callback(ep, epochs, float(avg_loss))
        _maybe_save_best(ep, float(avg_loss), val_loss)
        if scheduler is not None:
            scheduler.step()

    model = _export_model()
    setattr(model, "best_plm_loss", best_loss)
    setattr(model, "best_plm_val_loss", best_val_loss)
    setattr(model, "best_plm_epoch", best_epoch)
    setattr(model, "last_plm_loss", last_loss)
    setattr(model, "last_plm_val_loss", last_val_loss)
    return model


def add_potts_models(base: PottsModel, delta: PottsModel) -> PottsModel:
    if len(base.h) != len(delta.h):
        raise ValueError("Potts model sizes do not match.")
    if sorted(base.edges) != sorted(delta.edges):
        raise ValueError("Potts model edges do not match.")
    h = [base.h[r] + delta.h[r] for r in range(len(base.h))]
    J: Dict[Tuple[int, int], np.ndarray] = {}
    for edge in base.edges:
        J[edge] = base.J[edge] + delta.J[edge]
    return PottsModel(h=h, J=J, edges=list(base.edges))


def fit_potts_delta_pseudolikelihood_torch(
    base_model: PottsModel,
    labels: np.ndarray,
    *,
    l2: float = 0.0,
    lambda_h: float = 0.0,
    lambda_J: float = 0.0,
    lr: float = 1e-3,
    lr_min: float = 1e-3,
    lr_schedule: str = "cosine",
    epochs: int = 200,
    batch_size: int = 512,
    seed: int = 0,
    verbose: bool = True,
    report_init_loss: bool = True,
    progress_callback: "callable | None" = None,
    progress_every: int = 10,
    batch_progress_callback: "callable | None" = None,
    device: str | None = None,
) -> PottsModel:
    """
    Fit a sparse delta Potts model (Δh, ΔJ) on top of a frozen base model using pseudolikelihood.
    Regularization:
      l2: weight for elementwise L2 on delta parameters.
      lambda_h: group L2 on each residue field vector (sparsity).
      lambda_J: group Frobenius on each edge coupling matrix (sparsity).
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for delta pseudolikelihood fitting.") from e

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    if N != len(base_model.h):
        raise ValueError("Labels shape does not match base Potts model size.")

    edges = sorted((min(r, s), max(r, s)) for r, s in base_model.edges if r != s)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if verbose:
        print(f"[plm-delta] using device={torch_device}")

    neigh = [[] for _ in range(N)]
    for r, s in edges:
        neigh[r].append(s)
        neigh[s].append(r)

    torch.manual_seed(seed)
    base_h = [torch.tensor(-base_model.h[r], dtype=torch.float32, device=torch_device) for r in range(N)]
    base_J: Dict[str, "torch.Tensor"] = {}
    for r, s in edges:
        key = f"{r}_{s}"
        base_J[key] = torch.tensor(-base_model.J[(r, s)], dtype=torch.float32, device=torch_device)

    delta_h = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros(base_model.h[r].shape[0], dtype=torch.float32, device=torch_device))
        for r in range(N)
    ])
    delta_J = torch.nn.ParameterDict()
    for r, s in edges:
        key = f"{r}_{s}"
        delta_J[key] = torch.nn.Parameter(
            torch.zeros(base_model.J[(r, s)].shape, dtype=torch.float32, device=torch_device)
        )

    X = torch.tensor(labels, dtype=torch.long, device=torch_device)
    loss_fn = torch.nn.CrossEntropyLoss()

    opt = torch.optim.Adam(list(delta_h) + list(delta_J.values()), lr=lr, weight_decay=0.0)
    schedule = lr_schedule.lower() if lr_schedule else "none"
    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)
    elif schedule != "none":
        raise ValueError(f"Unknown lr_schedule={lr_schedule!r}")

    def _logits_for_residue(x_batch: "torch.Tensor", r: int) -> "torch.Tensor":
        B = x_batch.shape[0]
        logits = (base_h[r] + delta_h[r]).unsqueeze(0).expand(B, -1)
        for s in neigh[r]:
            rr, ss = (r, s) if r < s else (s, r)
            key = f"{rr}_{ss}"
            Jmat = base_J[key] + delta_J[key]
            xs = x_batch[:, s]
            if r < s:
                logits = logits + Jmat[:, xs].T
            else:
                logits = logits + Jmat[xs, :]
        return logits

    def _group_norm(t: "torch.Tensor") -> "torch.Tensor":
        return torch.sqrt(torch.sum(t * t) + 1e-12)

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
        print(f"[plm-delta] init avg_loss={init_loss:.6f}")

    for ep in range(1, epochs + 1):
        rng.shuffle(idx)
        total = 0.0
        nobs = 0
        for start in range(0, T, batch_size):
            bidx = idx[start:start + batch_size]
            xb = X[bidx]
            opt.zero_grad()

            loss = 0.0
            for r in range(N):
                logits = _logits_for_residue(xb, r)
                y = xb[:, r]
                loss = loss + loss_fn(logits, y)

            if l2 > 0:
                l2_sum = torch.tensor(0.0, device=torch_device)
                for r in range(N):
                    l2_sum = l2_sum + torch.sum(delta_h[r] ** 2)
                for key in delta_J:
                    l2_sum = l2_sum + torch.sum(delta_J[key] ** 2)
                loss = loss + l2 * l2_sum

            if lambda_h > 0:
                gh = torch.tensor(0.0, device=torch_device)
                for r in range(N):
                    gh = gh + _group_norm(delta_h[r])
                loss = loss + lambda_h * gh

            if lambda_J > 0:
                gJ = torch.tensor(0.0, device=torch_device)
                for key in delta_J:
                    gJ = gJ + _group_norm(delta_J[key])
                loss = loss + lambda_J * gJ

            loss.backward()
            opt.step()

            total += float(loss.item()) * len(bidx)
            nobs += len(bidx)
            if batch_progress_callback:
                batch_num = start // batch_size + 1
                batch_progress_callback(ep, epochs, batch_num, total_batches)

        avg_loss = total / max(1, nobs)
        if verbose and (ep == 1 or ep % max(1, epochs // 10) == 0):
            print(f"[plm-delta] epoch {ep:4d}/{epochs}  avg_loss={avg_loss:.6f}")
        if progress_callback and (ep == 1 or ep == epochs or ep % progress_every == 0):
            progress_callback(ep, epochs, float(avg_loss))
        if scheduler is not None:
            scheduler.step()

    h = [-hp.detach().cpu().numpy().astype(float) for hp in delta_h]
    J = {}
    for r, s in edges:
        key = f"{r}_{s}"
        J[(r, s)] = -delta_J[key].detach().cpu().numpy().astype(float)

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


def compute_pseudolikelihood_scores(
    model: PottsModel,
    labels: np.ndarray,
    *,
    batch_size: int = 512,
    device: str | None = None,
    return_residue_scores: bool = False,
    use_torch: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute per-frame pseudolikelihood scores for a fixed Potts model.

    Returns
    -------
    scores
        Array of shape (T,) with sum_r log p(x_r | x_-r) per frame.
    residue_scores
        Optional array of shape (T, N) with per-residue log-prob contributions.
    """
    labels = np.asarray(labels, dtype=int)
    T, N = labels.shape
    if N != len(model.h):
        raise ValueError("Labels shape does not match Potts model size.")
    if batch_size <= 0:
        batch_size = T

    edges = sorted((min(r, s), max(r, s)) for r, s in model.edges if r != s)
    neigh = [[] for _ in range(N)]
    for r, s in edges:
        neigh[r].append(s)
        neigh[s].append(r)

    if use_torch:
        try:
            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_device = torch.device(device)

            h_params = [
                torch.tensor(-model.h[r], dtype=torch.float32, device=torch_device)
                for r in range(N)
            ]
            J_params = {}
            for r, s in edges:
                key = f"{r}_{s}"
                J_params[key] = torch.tensor(-model.J[(r, s)], dtype=torch.float32, device=torch_device)

            X = torch.tensor(labels, dtype=torch.long, device=torch_device)

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

            scores = []
            residue_scores = [] if return_residue_scores else None
            with torch.no_grad():
                for start in range(0, T, batch_size):
                    xb = X[start:start + batch_size]
                    B = xb.shape[0]
                    frame_scores = torch.zeros(B, dtype=torch.float32, device=torch_device)
                    res_scores = None
                    if return_residue_scores:
                        res_scores = torch.zeros((B, N), dtype=torch.float32, device=torch_device)
                    for r in range(N):
                        logits = _logits_for_residue(xb, r)
                        log_probs = torch.log_softmax(logits, dim=1)
                        contrib = log_probs[torch.arange(B, device=torch_device), xb[:, r]]
                        frame_scores = frame_scores + contrib
                        if res_scores is not None:
                            res_scores[:, r] = contrib
                    scores.append(frame_scores.cpu().numpy())
                    if res_scores is not None and residue_scores is not None:
                        residue_scores.append(res_scores.cpu().numpy())

            score_arr = np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=float)
            res_arr = None
            if return_residue_scores and residue_scores is not None:
                res_arr = np.concatenate(residue_scores, axis=0) if residue_scores else np.zeros((0, N), dtype=float)
            return score_arr, res_arr
        except Exception:
            if use_torch:
                pass

    scores = []
    residue_scores = [] if return_residue_scores else None
    for start in range(0, T, batch_size):
        xb = labels[start:start + batch_size]
        B = xb.shape[0]
        frame_scores = np.zeros(B, dtype=float)
        res_scores = None
        if return_residue_scores:
            res_scores = np.zeros((B, N), dtype=float)
        for r in range(N):
            hr = model.h[r]
            logits = -hr[None, :].repeat(B, axis=0)
            for s in neigh[r]:
                if r < s:
                    Jmat = model.J[(r, s)]
                    xs = xb[:, s]
                    logits -= Jmat[:, xs].T
                else:
                    Jmat = model.J[(s, r)]
                    xs = xb[:, s]
                    logits -= Jmat[xs, :]
            max_logits = np.max(logits, axis=1, keepdims=True)
            logsumexp = max_logits + np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
            log_probs = logits - logsumexp
            contrib = log_probs[np.arange(B), xb[:, r]]
            frame_scores += contrib
            if res_scores is not None:
                res_scores[:, r] = contrib
        scores.append(frame_scores)
        if res_scores is not None and residue_scores is not None:
            residue_scores.append(res_scores)

    score_arr = np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=float)
    res_arr = None
    if return_residue_scores and residue_scores is not None:
        res_arr = np.concatenate(residue_scores, axis=0) if residue_scores else np.zeros((0, N), dtype=float)
    return score_arr, res_arr
