# AllosKin

AllosKin (Allostery + Kinetics) is a research framework for analyzing conformational dynamics and allosteric signaling in proteins. It combines descriptor extraction, metastable state discovery, residue clustering, and Potts-based sampling to compare ensembles and quantify state-specific signatures.

This repository is a monorepo with:

- `alloskin/`: core Python library (feature extraction, metastable analysis, clustering, Potts sampling).
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
   - `POST /api/v1/projects` with `{ "name": "My Project" }`.
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

Data is stored under the Docker volume mapped to `ALLOSKIN_DATA_ROOT` (default in compose: `/data/alloskin`).

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

## Repository Structure

```
alloskin/                Core library
backend/                 FastAPI + RQ workers
frontend/                React UI
docs/                    Documentation
```

## Notes

- The backend uses Redis to queue jobs. In Docker, Redis is started automatically.
- Results are persisted to `ALLOSKIN_DATA_ROOT/results` and referenced in run metadata.
- For API details, use the OpenAPI docs at `/docs`.
