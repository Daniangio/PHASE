#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"
if [ -d "${ROOT_DIR}/.venv-potts-fit" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
elif [ -d "${ROOT_DIR}/.venv" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv"
else
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
fi

prompt() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  if [ -z "$var" ]; then
    echo "$default"
  else
    echo "$var"
  fi
}

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  VENV_DIR="${VIRTUAL_ENV}"
  echo "Using active virtual environment at: ${VENV_DIR}"
else
  echo "No active virtual environment detected."
  if [ -x "${DEFAULT_ENV}/bin/python" ]; then
    echo "Activate it with: source ${DEFAULT_ENV}/bin/activate"
  else
    echo "Create one first: scripts/potts_setup.sh"
  fi
  exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

OFFLINE_ROOT=""
OFFLINE_PROJECT_ID=""
OFFLINE_SYSTEM_ID=""
MAX_CLUSTER_FRAMES=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id)
      OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id)
      OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --max-cluster-frames)
      MAX_CLUSTER_FRAMES="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

if [ -z "$OFFLINE_ROOT" ]; then
  offline_prompt_root "${ROOT_DIR}/data"
else
  OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"
  export PHASE_DATA_ROOT="$OFFLINE_ROOT"
fi

if [ -z "$OFFLINE_PROJECT_ID" ]; then
  offline_select_project
fi

if [ -z "$OFFLINE_SYSTEM_ID" ]; then
  offline_select_system
fi
STATE_ROWS="$(offline_select_analysis_states)"
if [ -z "$STATE_ROWS" ]; then
  echo "Select at least one state to cluster."
  exit 1
fi
STATE_IDS="$(printf "%s\n" "$STATE_ROWS" | awk -F'|' '{print $1}' | paste -sd, -)"
CLUSTER_NAME="$(prompt "Cluster name" "cluster_${OFFLINE_SYSTEM_ID}")"
N_JOBS="$(prompt "Worker processes (0 = all cpus)" "1")"
DENSITY_Z="$(prompt "Density z (auto or float)" "2.0")"
if [ -z "$MAX_CLUSTER_FRAMES" ]; then
  MAX_CLUSTER_FRAMES="$(prompt "Max cluster frames (blank = no limit)" "")"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.cluster_npz
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --state-ids "$STATE_IDS"
  --cluster-name "$CLUSTER_NAME"
  --n-jobs "$N_JOBS"
  --density-z "$DENSITY_Z"
)
if [ -n "$MAX_CLUSTER_FRAMES" ]; then
  CMD+=(--max-cluster-frames "$MAX_CLUSTER_FRAMES")
fi

echo "Running clustering..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"
