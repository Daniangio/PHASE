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

## Metadata Layout

Metadata is stored alongside the artifact it describes (system, clusters, models, samples).
Schemas are documented in `docs/metadata/`:
- `docs/metadata/system.md`
- `docs/metadata/descriptors.md`
- `docs/metadata/cluster.md`
- `docs/metadata/potts_model.md`
- `docs/metadata/sample.md`

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

## Recommended Pipeline

The recommended way to run PHASE is:

- keep one shared project tree on the host, for example `/scratch/$USER/phase-data`
- let Docker use that tree for the webserver
- let `phase_console` use that same tree locally

This is the best operational model because:

- there is no upload/download step between local CLI and webserver
- local CLI can run heavy jobs directly on the host
- the web UI can immediately visualize the same files
- metadata stays in one place

The rest of this README assumes that this shared-data workflow is your default setup.

See also:

- `docs/shared_data_root.md`
- `docs/docker_gpu.md`

## Shared-Root Setup

This section is the recommended setup to do once, then reuse every day.

### 1. Choose the shared host data root

Use one host directory for all PHASE projects, for example:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
mkdir -p "$PHASE_DATA_ROOT"
```

Alternatively, run `./scripts/phase_console.sh` and enter the desired data root at the first prompt.
The console stores that choice in repo-local defaults:

- `.phase_defaults`, used by PHASE helper scripts
- `.env`, read automatically by Docker Compose

The next `phase_console` run will offer the last selected root as the default.

You can verify the active shell value with:

```bash
echo "$PHASE_DATA_ROOT"
```

### 2. Make Docker write files as your user

Generate `.env` for Docker Compose:

```bash
./scripts/compose_env.sh
```

This writes:

```bash
COMPOSE_PROJECT_NAME=<unique docker compose project name>
PHASE_UID=<your uid>
PHASE_GID=<your gid>
PHASE_DOCKER_USER=<your uid>:<your gid>
PHASE_HOST_DATA_ROOT=<host data root mounted into Docker>
PHASE_DATA_ROOT=<your exported PHASE_DATA_ROOT>   # if set
PHASE_FRONTEND_PORT=<optional host frontend port>
PHASE_BACKEND_PORT=<optional host backend port>
PHASE_REDIS_PORT=<optional host redis port>
```

The compose file uses these values so `backend` and `worker` write files as your host user, not as `root`.
`COMPOSE_PROJECT_NAME` also prevents container/network name collisions when several users run PHASE from repos with the same directory name.

Important:

- Docker Compose uses `PHASE_HOST_DATA_ROOT` for the host bind mount; this avoids stale shell `PHASE_DATA_ROOT` values overriding `.env`
- `./scripts/compose_env.sh` reads root defaults from `PHASE_HOST_DATA_ROOT`, then `.phase_defaults`, then shell `PHASE_DATA_ROOT`, then `.env`, then `./data`
- `./scripts/compose_env.sh` preserves port values and `COMPOSE_PROJECT_NAME` already present in `.env`, unless you override them in the shell
- if you choose a root through `phase_console`, Docker will use the same root on the next `docker compose up`
- direct `docker compose up` without `.env` is allowed, but services run as `root`; run `./scripts/compose_env.sh` once to make Docker write as your user

### 3. Start the web stack

CPU-only:

```bash
docker compose up --build
```

With GPU access:

```bash
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build
```

### 4. Verify the permission model once

Inside the worker container:

```bash
docker compose exec worker id
```

You should see your host UID/GID, not `0:0`.

On the host:

```bash
find "$PHASE_DATA_ROOT" -maxdepth 2 ! -user "$(id -un)" | head
```

This should ideally return nothing.

### 5. If you already have a root-owned tree

If old Docker runs wrote files as `root`, repair ownership once:

```bash
sudo ./scripts/fix_shared_data_root.sh "$PHASE_DATA_ROOT"
./scripts/compose_env.sh
```

Then restart the stack.

## Normal Daily Usage

After the setup above, day-to-day usage should be simple.

### Start the webserver

CPU-only:

```bash
docker compose up -d
```

With GPU:

```bash
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
```

The web UI is then available at:

- `http://localhost:18080`

Default host ports are intentionally non-standard to avoid clashing with other local Docker web stacks:

- frontend: `18080`
- backend API: `18000`
- redis: `16380`

For multiple users on the same machine, each clone should use a different Compose project name and different host ports. Edit `.env` directly, for example:

```bash
COMPOSE_PROJECT_NAME=phase_raimoc
PHASE_FRONTEND_PORT=18180
PHASE_BACKEND_PORT=18100
PHASE_REDIS_PORT=16180
```

Then start normally:

```bash
./scripts/compose_env.sh
docker compose up -d
```

Replace the project name and port numbers with values unique to that user. The frontend is available at `http://localhost:<PHASE_FRONTEND_PORT>`.

### Development stack

If you explicitly want live reload for frontend/backend/worker, use the
development compose file:

```bash
docker compose -f docker-compose.yaml -f docker-compose.dev.yml up --build
```

That stack runs:

- `uvicorn --reload`
- CRA `react-scripts start`
- auto-restarting RQ worker

It is intentionally not the default. If you see webpack/eslint warnings in the
frontend container logs, you are running the development stack.

### Start `phase_console`

Because `PHASE_DATA_ROOT` is already exported in your shell, just run:

```bash
./scripts/phase_console.sh
```

`phase_console` will now operate by default on the same shared root used by the webserver.

### How to work normally

The intended workflow is:

1. create/manage projects and systems in the web UI or in `phase_console`
2. run heavy fits, sampling, or multiprocessing tasks locally with `phase_console`
3. open the web UI to inspect the exact same outputs
4. alternate freely between local CLI and webserver, because both are reading and writing the same project tree

### What path each side should use

Use:

- local CLI: `PHASE_DATA_ROOT=/scratch/$USER/phase-data`
- inside Docker: `/data/phase`

This is expected and correct. They point to the same files through the Docker bind mount.

### What not to do

- do not run `phase_console` with `sudo`
- do not point local CLI tools at `/data/phase`
- do not keep a separate local `./data` tree if your real projects are in `/scratch/$USER/phase-data`
- do not start Docker before running `./scripts/compose_env.sh` at least once

## Quick Start Summary

One-time setup:

```bash
echo 'export PHASE_DATA_ROOT=/scratch/$USER/phase-data' >> ~/.bashrc  # or ~/.zshrc
source ~/.bashrc  # or source ~/.zshrc
mkdir -p "$PHASE_DATA_ROOT"
./scripts/compose_env.sh
docker compose up --build
```

Normal use:

```bash
docker compose up -d
./scripts/phase_console.sh
```

Normal use with GPU web jobs:

```bash
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
./scripts/phase_console.sh
```

## Running with Docker

Requirements: Docker + Docker Compose.

```bash
# Build and start all services
docker compose up --build
```

If you want Docker containers to use host GPUs for Potts fitting, see `docs/docker_gpu.md`.

Services:
- Backend API: `http://localhost:18000` (OpenAPI docs at `/docs`)
- Frontend: `http://localhost:18080`

Data is stored under the Docker volume mapped to `PHASE_DATA_ROOT` inside the container (default in compose: `/data/phase`).

## Shared Data Root

This is the recommended setup. The full operational guide is in `docs/shared_data_root.md`.

### Multiple Workers

To enable parallel background jobs (including fan-out clustering), scale the worker service:

```bash
# Example: 4 worker processes
docker compose up --build --scale worker=4
```

Note: more workers increases CPU and memory usage. If you only run one worker, the clustering job will fall back to a single-process path.

## Running Locally (Without Docker)

Python 3.11+ recommended.

This mode is mainly useful for development of the backend/frontend themselves.
For normal project work, prefer the shared-data workflow above instead of a fully separate local-only tree.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r backend/requirements.txt
uv pip install -e .

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
