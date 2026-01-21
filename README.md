# PHASE

PHASE (Protein Hamiltonian for Annealed Sampling of conformational Ensembles) is a modular framework for learning reduced Hamiltonians from molecular dynamics trajectories and generating novel protein conformations via calibrated annealed sampling. It bridges analysis and generation by combining trajectory preprocessing, descriptor extraction, metastable state discovery, residue-level clustering, and Potts-based sampling into a reproducible pipeline.

This repository contains:

- `phase/`: core Python library (feature extraction, metastable analysis, clustering, Potts sampling).
- `backend/`: FastAPI server + RQ workers for background jobs.
- `frontend/`: React UI for project and results management.
- `docs/`: method notes and architecture documents.

## Core Concepts

- **Project**: top-level workspace that groups multiple systems.
- **System**: a set of macro-states (PDB + trajectory per state) with stored descriptors.
- **Descriptors**: per-residue dihedral features saved as NPZ for reuse.
- **Metastable states**: optional TICA/MSM-based refinement of macro-states.
- **Cluster NPZ**: per-residue clustering results used by Potts sampling.
- **Analyses**:
  - Static reporters (information imbalance, per-residue signals).
  - Potts sampling + replica exchange / SA-QUBO.

## Typical Workflow (Web Server)

1. **Create a project**
   - Decide project name and add optional description.
2. **Create a system**
   - Upload PDBs for the macro-states (multipart form).
3. **Upload trajectories + build descriptors**
   - Upload trajectories per state; descriptors are built and stored on disk.
4. **(Optional) Run metastable discovery**
   - Run TICA/MSM to compute metastable states.
5. **Run residue clustering**
   - Generates a Cluster NPZ for Potts sampling.
6. **Run analysis**
   - Static reporters or Potts sampling jobs from the UI.
7. **Visualize results**
   - Use the frontend to explore plots and download artifacts.

Background jobs (metastable discovery, clustering, Potts sampling) are executed by RQ workers. See `docs/clustering_architecture.md` for the clustering fan-out flow.

## Running with Docker (Recommended)

Requirements: Docker + Docker Compose.

```bash
# Build and start all services
docker compose up --build
```

Services:
- Backend API: `http://localhost:8000` (OpenAPI docs at `/docs`)
- Frontend: `http://localhost:3000`

Data is stored under the Docker volume mapped to `PHASE_DATA_ROOT` (default in compose: `/data/phase`).

### Multiple Workers

To enable parallel background jobs (including fan-out clustering), scale the worker service:

```bash
# Example: 4 worker processes
docker compose up --build --scale worker=4
```

Note: more workers increases CPU and memory usage. If you only run one worker, the clustering job will fall back to a single-process path.

## Running Locally (Without Docker)

Python 3.11+ recommended.

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -e .

# Backend
uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm start
```

## Local Potts Model Fitting

You can fit Potts models on a separate machine (e.g., with CUDA) and upload the
`potts_model.npz` back to the web UI.

1) Create a dedicated uv environment once:

```bash
./scripts/potts_setup.sh
source .venv-potts-fit/bin/activate
```

2) Run the interactive fitter (requires an active venv):

```bash
./scripts/potts_fit.sh
```

The script prompts for the input Cluster NPZ, PLM hyperparameters, and device
(`auto`, `cuda`, or `cpu`). The fitted model is saved as `potts_model.npz` in the
chosen results directory.

## Notes

- The backend uses Redis to queue jobs. In Docker, Redis is started automatically.
- Results are persisted to `PHASE_DATA_ROOT/results` and referenced in run metadata.
- For API details, use the OpenAPI docs at `/docs`.
- Potts models can be fit once and reused for sampling. See `docs/potts_overview.md` for CLI examples.
