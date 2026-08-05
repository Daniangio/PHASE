# PHASE Framework: Spectral Sector Analysis and Pharmacological Profiling

## 1. Theoretical Foundation: Beyond 1D Energy Projections

Historically, receptor activation was modeled as a two-state system, projecting ligand efficacy onto a 1D energy difference ($\Delta E$). Within the PHASE Potts model framework, this projection assumes the form:
$$\Delta E(\mathbf{s}) = \mathcal{H}_{\text{inactive}}(\mathbf{s}) - \mathcal{H}_{\text{active}}(\mathbf{s})$$

**The Thermodynamic Flaw:** This approach is fundamentally unsuitable for highly frustrated systems like GPCRs and fails to capture biased signaling. Biased ligands do not occupy a fractional coordinate on a linear inactive-active axis; they stabilize orthogonal conformational microstates. Furthermore, short ligand-bound Molecular Dynamics (MD) trajectories projected onto models fit strictly on reference states (e.g., APO) will suffer from basis set incompleteness, artificially forcing the trajectory "back" into known energy basins due to extreme energetic penalties on unobserved states.

To rigorously evaluate biased signaling and allostery, the analytical framework must shift from global energy states to the statistical mechanics of **discrete probability distributions** and **allosteric coupling networks**.

---

## 2. Spectral Sector Analysis: Isolating the Signal

Proteins are organized into "sectors"—sparse, contiguous networks of residues that exhibit highly correlated fluctuations. To extract these dynamic sectors from the discrete microstate sampling of the PHASE framework, we analyze the pairwise coupling matrices ($J_{ij}$).

The framework must exclusively use the **zero-sum gauge** (Ising gauge) to ensure matrix norms are independent of the arbitrary fitting gauge:
$$\sum_{s_i} J_{ij}(s_i, s_j) = 0 \quad \text{and} \quad \sum_{s_j} J_{ij}(s_i, s_j) = 0$$

The $q \times q$ discrete interaction space is compressed into a scalar spatial matrix using the Frobenius norm:
$$F_{ij} = \sqrt{\sum_{s_i} \sum_{s_j} J_{ij}(s_i, s_j)^2}$$

### Mode A: Structural Scaffolds (Single-State Entropy)
Diagonalizing a single-state matrix ($F_{\text{single}}$) identifies the axes of maximum absolute coupling variance. Because highly flexible regions (e.g., ICL3, ECL2) sample vast configurational entropy, their couplings dominate this space.
* **Physical Meaning:** Identifies the rigid architectural blocks and highly entropic boundaries of a specific thermodynamic well (the structural scaffold).

### Mode B: Functional Pathways (Differential Laplacian)
Allostery is an intrinsically differential phenomenon. To suppress the "hub" effect of intrinsically highly coupled flexible loops, we compute the **Signed Normalized Laplacian** ($\mathcal{L}$) of the difference matrix $\Delta F = F_{\text{State B}} - F_{\text{State A}}$.

1. Compute absolute degree: $D_{ii} = \sum_{j} |\Delta F_{ij}|$
2. Construct Normalized Laplacian:
$$\mathcal{L}_{ij} = \begin{cases} 
1 & \text{if } i = j \text{ and } D_{ii} > 0 \\
-\frac{\Delta F_{ij}}{\sqrt{D_{ii} D_{jj}}} & \text{if } i \neq j \text{ and } D_{ii}, D_{jj} > 0 \\
0 & \text{otherwise}
\end{cases}$$
* **Physical Meaning:** The Fiedler eigenvectors (corresponding to the smallest non-zero eigenvalues) of $\mathcal{L}$ isolate the sparse, rigid allosteric networks that specifically exchange coupling energy to facilitate macroscopic state transitions.

---

## 3. Community Detection and the Thermodynamic Noise Floor

To automatically partition the protein into discrete communities, residues are projected into a low-dimensional eigenspace defined by the $k$ Fiedler vectors, row-normalized to a unit hypersphere (matrix $Y$).

### Density Peak Clustering (DPC)
We apply `dadapy` to the hyperspherical coordinates, mandating an angular distance metric (e.g., **Cosine distance**) to prevent coordinate distortion. 

### Core-Halo Topological Filtration
In highly frustrated systems, the majority of residues form an inert structural bulk that absorbs thermal noise. Row-normalization artificially amplifies this noise. We strictly filter the thermodynamic bulk using `dadapy`'s saddle-point topology:
* **Core:** Residues with local density strictly greater than the bounding saddle point. These are statistically significant functional hubs.
* **Halo:** Residues mathematically assigned to a peak, but with density lower than the saddle point. These represent background thermal noise and must be discarded from functional definitions.

---

## 4. The Allosteric Piston Model

By combining the structural clustering (Mode A) and the functional rewiring clustering (Mode B), we can mathematically assign distinct physical roles to every residue via set-theoretic intersection.

Let $C_{\text{struct}}$ be the Single-State core assignments and $C_{\text{func}}$ be the $\Delta F$ core assignments. 

| Classification | Mathematical Signature | Physical Interpretation |
| :--- | :--- | :--- |
| **Allosteric Piston** | $\text{Core}_{\text{struct}} \land \text{Core}_{\text{func}}$ | A highly cohesive structural unit that rewires collectively. A true mechanical gear. |
| **Structural Scaffold** | $\text{Core}_{\text{struct}} \land (\text{Halo}_{\text{func}} \lor \text{Unassigned})$ | Rigid architectural blocks deaf to the activation signal. |
| **Transient Switches** | $(\text{Halo}_{\text{struct}} \lor \text{Unassigned}) \land \text{Core}_{\text{func}}$ | Isolated residues whose correlations spike transiently to bridge the allosteric network. |
| **Thermodynamic Bulk** | $\text{Halo}_{\text{struct}} \land \text{Halo}_{\text{func}}$ | Background noise; fluctuating strictly within the baseline thermal bath. |

---

## 5. Ligand Profiling via Masked Empirical Projections

To evaluate short screening MDs without fitting noise-dominated Hamiltonians, we project the empirical dynamic flow of the short trajectory onto the predefined reference pathways.

### The Empirical Flow Matrix
Because metrics like Information Imbalance are mathematically unsuitable for comparing these distinct multi-state probability distributions, we rigorously quantify the dynamic correlations using the empirical **Mutual Information** matrix ($M^{\text{short}}$) derived from the categorical microstate frequencies:
$$M_{ij}^{\text{short}} = \sum_{s_i} \sum_{s_j} P(s_i, s_j) \ln \left( \frac{P(s_i, s_j)}{P(s_i)P(s_j)} \right)$$

*(Alternatively, Kullback-Leibler Divergence can be used to compare marginal distributions against the reference state).*

### Masked Piston Projection
Global projection accumulates thermal noise from the bulk scaffold. We must project $M^{\text{short}}$ exclusively onto the $K$ discrete Allosteric Pistons identified in Section 4. 

For a Piston $k$ containing residue indices $\Omega_k$, define a masked eigenvector $\mathbf{\tilde{v}}^{(k)}$:
$$\tilde{v}_i^{(k)} = \begin{cases} v_i & \text{if } i \in \Omega_k \\ 0 & \text{if } i \notin \Omega_k \end{cases}$$

The Piston-Specific Commitment Score is:
$$\mathcal{P}_k = \left( \mathbf{\tilde{v}}^{(k)} \right)^T M^{\text{short}} \mathbf{\tilde{v}}^{(k)} = \sum_{i \in \Omega_k} \sum_{j \in \Omega_k} v_i M^{\text{short}}_{ij} v_j$$

### Pharmacological Fingerprinting

By selecting specific reference states, we dictate the thermodynamic question being asked of the ligand:

1.  **Full Agonist:**
    * *Reference:* $F_{\text{act-G}}$ and $\Delta F = F_{\text{act-G}} - F_{\text{inact-APO}}$.
    * *Signature:* Rapid, high Commitment Scores ($\mathcal{P}$) on canonical activation pistons.
2.  **Biased Agonist:**
    * *Reference:* Project onto both G-protein pistons ($\Omega_{\text{G}}$) and $\beta$-arrestin pistons ($\Omega_{\text{B}}$) independently.
    * *Signature:* Asymmetric commitment (e.g., $\mathcal{P}_{\text{B}} \gg \mathcal{P}_{\text{G}}$).
3.  **Inverse Agonist:**
    * *Reference:* $F_{\text{inact-APO}}$ and $\Delta F = F_{\text{inact-ZMA}} - F_{\text{inact-APO}}$.
    * *Signature:* High commitment on the sparse "lockdown" tracks that clamp the receptor.
4.  **Neutral Antagonist:**
    * *Signature:* Near-zero commitment scores across all functional tracks; maintains background thermal noise without organizing directed currents.

### The "Origin Scaffold" Projection (High Sensitivity for Short MDs)
If the Single State is set to the **Inactive APO** ($F_{\text{inact-APO}}$) while maintaining the activation $\Delta F$, the pistons represent the "Origin Scaffold"—the rigid locks that maintain the basal state. 
When an agonist binds, the local strain shatters these preexisting correlations immediately. Therefore, the signature of a full agonist in this specific setup is a **massive drop** in $\mathcal{P}_k$ relative to an APO baseline simulation, offering extremely high temporal sensitivity for microsecond screening.