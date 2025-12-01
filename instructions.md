0. High-level pipeline
For each system (PDB + trajectories):
Feature extraction (already implemented)


For each frame and residue: compute
 ([ \sin\phi, \cos\phi, \sin\psi, \cos\psi, \sin\chi_1, \cos\chi_1 ]).


Store per-trajectory as descriptors_{trajectory_id}.npz.


Metastable state analysis (triggered on trajectory upload)


Pool all trajectories of the system.


Run TICA on concatenated features.


Cluster into microstates (K_micro, default 20).


Cluster microstates into metastable states (K_meta chosen by silhouette, k=1–4).


For each trajectory: save per-frame metastable labels and per-state representative PDBs.


Analysis 1: MI + QUBO (system-level job)


Using all trajectories and metastable labels:


For each metastable-state pair (a,b):
 compute MI(X_i, S∈{a,b}) for each residue (continuous MI via scikit-learn).
 pre-filter top K residues by MI for that pair.


Compute global redundancy matrix between residues.


Solve a QUBO per (a,b) to pick a minimal, non-redundant residue set.


Save: MI matrix, redundancy matrix, QUBO-selected residues per pair.


Analysis 2: microswitch discovery (system-level job)


Using QUBO-selected residues:


For each state pair (a,b) and each seed residue:
 build a local neighborhood graph and expand to a microswitch.


Save: microswitch groups and stats.


Expose for visualization: select pair (a,b) → highlight microswitches on the representative PDBs.



1. Data model (conceptual)
Project


id, name, owner


System


id, project_id


pdb_file_path


trajectories: list of Trajectory IDs


Trajectory


id, system_id


source_state_label (optional user label)


topology_path, trajectory_path


descriptors_path (descriptors_{trajectory_id}.npz)


metastable_labels_path (meta_labels_{trajectory_id}.npy)


MetastableStateModel (per system)


tica_model_path


microstate_kmeans_path


microstate_assignments (not persisted long-term if you prefer)


metastable_assignments_per_microstate (mapping microstate → metastable)


n_microstates (K_micro)


n_metastable_states (K_meta)


representative_pdb_paths (one per metastable state)


MIQUBOResults (per system)


metastable_pairs: list of (a, b)


mi_per_pair_per_residue: e.g. mi[a,b][res_id] = float


redundancy_matrix_path (global residue–residue redundancy)


qubo_selected_residues: selected[a,b] = [res_ids]


MicroswitchResults (per system)


microswitches[a,b] = list of:


id


seed_residue_id


member_residue_ids


basic annotations (e.g. distance change, heuristic label)


stats (mean feature change, etc.)



2. API / service endpoints
You can map this to REST routes or background jobs; I’ll describe them as logical services.
2.1 Trajectory upload and feature computation (already there)
POST /systems/{system_id}/trajectories
Input:


PDB/trajectory file references


Steps:


Store files, create Trajectory record.


Compute sin/cos dihedral descriptors for each frame/residue.


Save to descriptors_{trajectory_id}.npz.


Call MetastableStateService.recompute(system_id) (see below).


Output:


trajectory_id


status of metastable recomputation job



2.2 Metastable state recomputation
POST /systems/{system_id}/metastable/recompute
 (Internally called on upload; you can also expose it explicitly.)
Input
system_id


Processing
Load features for all trajectories

 For each trajectory in system:


Load descriptors_{trajectory_id}.npz:


Shape: (n_frames, n_residues, 6)


Optionally subsample frames if extremely large.


Concatenate over trajectories →
 X_all: shape (N_total_frames, n_residues * 6).


TICA on concatenated features


Standardize each feature column of X_all.


Fit TICA model with lag tau (config).


Transform: Z_all = TICA.transform(X_all) → shape (N_total_frames, d).


Microstate clustering


Hyperparameter: K_micro (default 20, small).


Run k-means on Z_all with K_micro.


Assign each frame a microstate label m_t ∈ {0,...,K_micro-1}.


Metastable state clustering (1–4 clusters via silhouette)


Compute microstate centroids in TICA space:
 C_k = mean(Z_all[m_t == k]).


For k_meta in {1,2,3,4}:


Cluster {C_k} into k_meta clusters (e.g. k-means).


Compute silhouette score on microstate centroids.


Choose k_meta with highest silhouette score.


If k_meta = 1 is allowed, special-case silhouette; or fall back to k_meta=1 if all >1 are very poor.


This defines a mapping:


microstate k → metastable state s_k ∈ {0,...,K_meta-1}.


Per-trajectory metastable labels + representative structures

 For each trajectory:


Using stored microstate assignment indices, derive S_t = s_{m_t} per frame.


Save meta_labels_{trajectory_id}.npy (int array).


For each metastable state s:


Collect frames in that trajectory with S_t = s.


Pick the frame whose TICA coordinates are closest to the mean TICA vector of that state (over all trajectories).


Extract coordinates and save a representative PDB for state s (e.g. state_{s}.pdb).


Persist MetastableStateModel


Save TICA model, k-means microstate model, microstate → metastable mapping, K_micro, K_meta, representative PDBs.


Output
n_metastable_states


For each trajectory: path to metastable labels file


Paths to representative PDB files per metastable state



2.3 Analysis 1: MI and QUBO
POST /systems/{system_id}/analysis/mi_qubo
Runs on the current metastable model and trajectories.
Input
system_id


Optional:


min_frames_per_state (default ~500)


top_k_per_pair (e.g. 100)


lambda_redundancy for QUBO


Processing
Load labels and features

 For each trajectory:


Load meta_labels_{trajectory_id}.npy → S_t.


Load descriptors_{trajectory_id}.npz → features (n_frames, n_residues, 6).


Keep a mapping from “global frame index” to (trajectory_id, local_frame_index).


Enumerate metastable state pairs


Let metastable states be S ∈ {0,...,K_meta-1}.


Build list of unordered pairs (a,b) with a<b.


Compute MI per pair and residue

 For each pair (a,b):


Collect all frames across all trajectories with S_t ∈ {a,b}.


Count frames per state: N_a, N_b.


If N_a < min_frames_per_state or N_b < min_frames_per_state:


Skip this pair (store as “insufficient data”).


Subsample from the majority state so that N_a ≈ N_b (balanced).


Now for this pair:


Define Y_ab as binary labels (0 for state a, 1 for state b).


For each residue i:


Extract features X_i for those frames:


shape: (N_samples, 6) (sin/cos φ, ψ, χ1).


Flatten per-residue features into (N_samples, 6) as input to scikit-learn:


Use mutual_info_classif(X_i, Y_ab, discrete_features=False).


This returns MI per feature dimension [MI_1,...,MI_6].


Aggregate to a single scalar for the residue, e.g.:


I_i^{(a,b)} = mean(MI_1,...,MI_6) or sum(MI_j).


Store result in mi[a,b][i] = I_i^{(a,b)}.


Global redundancy matrix

 Redundancy between residues i and j is approximated from continuous features:


Option 1 (simple, scalable):
 compute correlation-based redundancy.


Sample a subset of frames from all trajectories.


For each residue i, flatten its 6-d features over sampled frames → vector v_i.


For each pair (i,j):


Compute R_ij = |corr(v_i, v_j)| (absolute Pearson correlation).


Save redundancy matrix R (n_res x n_res).


QUBO per pair

 For each pair (a,b) with MI computed:


Let I_i = I_i^{(a,b)}.


Pre-filter residues: keep only top K = top_k_per_pair residues ranked by I_i.


Let candidate set indices be C_ab = {i_1,...,i_K}.


Build QUBO variables x_k ∈ {0,1} for each candidate residue k ∈ C_ab.


QUBO objective:
 [
 F^{(a,b)}(x) = \sum_{i \in C_{ab}} w_i x_i - \lambda \sum_{i<j, i,j \in C_{ab}} R_{ij} x_i x_j
 ]
 where:


w_i = I_i^{(a,b)}.


R_ij from redundancy matrix.


Implementation:


Construct Q matrix with:


Q_ii = -w_i (since we usually minimize in QUBO form)


Q_ij = λ R_ij for i≠j


Use a QUBO solver (classical) to find x minimizing xᵀ Q x.


Selected residues = { i in C_ab | x_i = 1 }.


Store qubo_selected_residues[a,b].


Persist MIQUBOResults


Save MI per pair per residue, redundancy matrix, and selected residues per pair.


Output
List of metastable pairs analyzed.


For each pair:


Selected residue IDs


Threshold used (top_k)


Location of stored MI and redundancy data.



2.4 Analysis 2: Microswitch discovery
POST /systems/{system_id}/analysis/microswitches
Uses QUBO-selected residues to identify microswitch groups.
Input
system_id


Optional:


distance cutoff for neighborhood (default 5 Å on CA–CA)


minimum MI threshold to consider a residue as seed (can reuse QUBO-selected set)


method for graph clustering (connected components vs community detection)


Processing
Load needed data


Load MetastableStateModel (TICA, clusters, centroids, representative PDBs).


Load all meta_labels_{trajectory_id}.npy.


Load all descriptors_{trajectory_id}.npz.


Load MIQUBOResults (selected residues per pair).


Load original PDB/topology to compute distances.


For each metastable pair (a,b)

 For each pair (a,b) with selected residues:


For each seed residue i in qubo_selected_residues[a,b]:

 2.1 Representative frames for states a and b


Identify TICA-space centroids for states a and b.


For each state s∈{a,b}:


Among all frames with S_t = s, pick frame closest to state centroid in TICA space.


Let those be frames f_a and f_b.


2.2 Local neighborhood N_i^{(a,b)}


In frame f_a, compute CA–CA distances from residue i to all other residues.


In frame f_b, do the same.


Define neighborhood:
 [
 N_i^{(a,b)} = { j \mid d_{ij}(f_a) \leq d_\text{cutoff} ;\text{or}; d_{ij}(f_b) \leq d_\text{cutoff} }
 ]
 with d_cutoff ~ 5 Å.


2.3 Local co-switching graph


Restrict to frames with S_t ∈ {a,b}.


For residues in N_i^{(a,b)}, extract their 6D features over these frames.


Define a co-switching similarity between residues j,k in N_i^{(a,b)}:


e.g. absolute correlation of their feature vectors over the restricted frames:
 [
 G_{jk}^{(a,b)} = |\text{corr}(f_j, f_k)|
 ]


Build an undirected graph:


nodes = residues in N_i^{(a,b)}


edges where G_{jk}^{(a,b)} ≥ threshold (e.g. 0.6).


2.4 Microswitch group


Compute the connected component containing the seed i (or use community detection if richer structure is desired).


Define this component as microswitch M_{i}^{(a,b)}.


Store:


seed_residue_id = i


member_residue_ids = component nodes


basic statistics:


mean feature difference between states a and b for each member residue


distance changes between key side-chain atoms (optional)


Persist MicroswitchResults


For each pair (a,b), list of microswitch groups with metadata.


Output
For each (a,b):


list of microswitch IDs


members per microswitch


simple metrics (e.g. average feature change, size)



2.5 Visualization endpoints
You already have an embedded PDB viewer. You can expose:
GET /systems/{system_id}/metastable_states
Returns:


number of metastable states


representative PDB path per state


basic stats (population, dwell time)


GET /systems/{system_id}/microswitches/{a}/{b}
Input:


a, b: metastable state indices


Returns:


representative PDB paths for states a and b


list of microswitches for (a,b):


seed residue index


member residue indices


optional label / classification (e.g. “ionic-lock-like” if your heuristic layer is implemented)


per-residue MI for this pair (useful to color by MI intensity)


QUBO-selected residues


Client can then:
Colour residues by membership (microswitch id) and MI value.


Toggle between PDB of state a and state b.


Show small panels summarizing which microswitches are involved in which transitions.



This is a coherent v1: it respects your choices (continuous MI, silhouette-based metastable count, top-K per pair) and is implementable in a modular way on a webserver. If you’d like, next step I can draft concrete Python pseudo-code for the two main services:
MetastableStateService.recompute(system_id)


AnalysisService.run_mi_qubo(system_id) and run_microswitches(system_id)