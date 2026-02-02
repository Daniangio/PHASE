# PMI and PLM in Potts models

This note explains the two fitting strategies used in PHASE: PMI (a fast heuristic)
and PLM (pseudolikelihood maximization).

## PMI (pointwise mutual information)
PMI uses empirical single-site and pairwise frequencies to build a quick initializer:

- Single-site fields:
  h_r(k) = -log p_r(k)
- Pairwise couplings:
  J_rs(k,l) = -log( p_rs(k,l) / (p_r(k) p_s(l)) )

In practice we add a small epsilon to probabilities and optionally center h/J
to remove constant offsets. PMI is fast and useful for quick baselines or as an
initializer, but it is not a maximum-likelihood fit.

## PLM (pseudolikelihood maximization)
PLM maximizes the sum of conditional log-likelihoods:

sum_r log P(x_r | x_-r; h, J)

For each residue r, the conditional distribution is a softmax over its states:

logits_r(k) = h_r(k) + sum_{s in neighbors(r)} J_rs(k, x_s)

This avoids the full partition function and is robust for larger systems.

## Implementation sketch (PHASE)
The PLM fit in `phase/potts/potts_model.py` is a symmetric global fit:

1) Build neighbor lists from the contact edges.
2) Initialize h and J from PMI (optional).
3) Use PyTorch to optimize the negative pseudolikelihood:
   - Sample mini-batches of frames.
   - For each residue r, compute logits_r for the batch.
   - Apply log-softmax and accumulate cross-entropy against the true labels.
   - Add L2 regularization on h and J.
4) Use a cosine or fixed learning-rate schedule, with optional progress logging.

This approach avoids the "fit each residue separately then symmetrize" pattern,
and directly optimizes a single set of symmetric couplings.
