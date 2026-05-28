# Shared Data Root

This document describes the clean way to make Docker services and local CLI tools
work on the same PHASE dataset tree.

## Goal

Use one host directory as the single PHASE data root, for example:

```bash
/scratch/$USER/phase-data
```

Docker mounts that directory into containers as:

```bash
/data/phase
```

Local CLI tools should point `PHASE_DATA_ROOT` to the host path, while Docker
services use the mounted container path.

## Recommended operational model

Use the shared data root as the default working mode for PHASE.

- Docker webserver:
  - project/system management
  - visualization
  - background jobs you want to launch from the UI
- Local CLI:
  - `phase_console`
  - heavy multiprocessing jobs
  - GPU jobs on the host
  - debugging and ad hoc scripts

Both sides should operate on the same on-disk project tree.

That gives you:

- no upload/download step between local and webserver
- one source of truth for metadata and artifacts
- local performance for heavy jobs
- immediate visibility in the web UI once outputs are written

The practical rule is simple:

- always set local `PHASE_DATA_ROOT` to the host path, for example `/scratch/$USER/phase-data`
- always run Docker with that same directory mounted as `/data/phase`

## Clean setup for a new repo

1. Pick a host data root:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
mkdir -p "$PHASE_DATA_ROOT"
```

2. Write your UID/GID into `.env` so Docker services run as your user:

```bash
./scripts/compose_env.sh
```

This writes:

```bash
PHASE_UID=<your uid>
PHASE_GID=<your gid>
PHASE_DOCKER_USER=<your uid>:<your gid>
PHASE_DATA_ROOT=<your exported PHASE_DATA_ROOT>   # if set
PHASE_FRONTEND_PORT=<optional host frontend port>
PHASE_BACKEND_PORT=<optional host backend port>
PHASE_REDIS_PORT=<optional host redis port>
```

PHASE also keeps a repo-local `.phase_defaults` file. `phase_console` updates this file whenever you choose a different data root at startup. `./scripts/compose_env.sh` reads roots in this order:

1. exported `PHASE_DATA_ROOT`
2. `.phase_defaults`
3. `.env`
4. `./data`

Docker Compose reads `.env` automatically, so after changing the root in `phase_console`, the webserver uses the same root on the next `docker compose up`.

If `.env` is missing, direct `docker compose up` falls back to running backend/worker as `root` to avoid startup permission errors. This is only a safety fallback. Run `./scripts/compose_env.sh` once so Docker writes as your host user.

When Docker is configured with a host data root that is not writable by the configured container user, the backend can fail on startup with:

```text
PermissionError: [Errno 13] Permission denied: '/data/phase/projects'
```

So the correct sequence is always:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
mkdir -p "$PHASE_DATA_ROOT"
./scripts/compose_env.sh
docker compose up --build
```

3. Start the stack:

```bash
docker compose up --build
```

With the current compose file, `backend` and `worker` run as
`PHASE_DOCKER_USER`, so files created inside Docker are owned by your host
user and remain writable from `phase_console`.

4. Run local CLI tools against the same data root:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/phase_console.sh
```

## Recommended startup patterns

### CPU-only shared workflow

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/compose_env.sh
docker compose up --build
```

Then in another shell:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/phase_console.sh
```

### Explicit development stack

If you need live reload while developing PHASE itself, start the development
stack explicitly:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
docker compose -f docker-compose.yaml -f docker-compose.dev.yml up --build
```

That stack is separate on purpose. It runs the CRA frontend dev server and will
log webpack/eslint warnings, which is expected in development mode.

### Shared workflow with Docker GPU access

If you also want webserver jobs to use GPUs:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/compose_env.sh
PHASE_GPU_CDI_DEVICE=nvidia.com/gpu=0 \
docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up --build
```

Then in another shell:

```bash
export PHASE_DATA_ROOT=/scratch/$USER/phase-data
./scripts/phase_console.sh
```

For GPU-specific details and validation, see `docs/docker_gpu.md`.

## Patching an existing root-owned data tree

If the Docker stack was previously run as `root`, files under
`/scratch/$USER/phase-data` may be owned by `root:root`. In that case local CLI
tools can read them but often cannot modify them without `sudo`.

The clean fix is:

1. Stop the Docker stack.
2. Rewrite ownership once.
3. Restart Docker with UID/GID mapping enabled.

### One-time ownership repair

Use the helper script:

```bash
sudo ./scripts/fix_shared_data_root.sh /scratch/$USER/phase-data
```

The script:

- changes ownership recursively to your current user
- makes directories traversable and writable by owner/group
- makes files writable by owner/group

If you need to override the target owner explicitly:

```bash
sudo TARGET_UID=$(id -u) TARGET_GID=$(id -g) \
  ./scripts/fix_shared_data_root.sh /scratch/$USER/phase-data
```

### After the repair

Regenerate `.env` and restart:

```bash
./scripts/compose_env.sh
docker compose up --build
```

After this, newly created project files should remain editable from both Docker
and local CLI tools without `sudo`.

## Important distinction: host path vs container path

Use:

- local CLI: `PHASE_DATA_ROOT=/scratch/$USER/phase-data`
- inside Docker: `PHASE_DATA_ROOT=/data/phase`

Do not point local CLI tools at `/data/phase`; that path only exists inside the
container.

## Sanity checks

Check ownership on the host:

```bash
ls -ld /scratch/$USER/phase-data
find /scratch/$USER/phase-data -maxdepth 2 ! -user "$(id -un)" | head
```

Check the effective user inside the worker container:

```bash
docker compose exec worker id
```

You should see your host UID/GID, not `0:0`.

## Failure mode to watch for

If files become root-owned again, one of these is true:

- the stack was started without the `.env` file generated by `./scripts/compose_env.sh`
- a different compose file is being used that overrides `user:`
- a one-off command was run in Docker as `root`

In that case, fix ownership again and restart the stack with the correct compose
configuration.

## What not to do

- Do not keep a separate `./data` tree for local CLI if the webserver uses `/scratch/$USER/phase-data`
- Do not run local CLI against `/data/phase`; that path exists only inside containers
- Do not run `phase_console` with `sudo`
- Do not start Docker before running `./scripts/compose_env.sh`, otherwise new files may be written with the wrong ownership
