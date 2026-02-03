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

DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"

trim() {
  printf "%s" "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

prompt() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  if [ -z "${var}" ]; then
    echo "${default}"
  else
    echo "${var}"
  fi
}

prompt_bool() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  var="${var:-$default}"
  var="$(printf "%s" "$var" | tr '[:upper:]' '[:lower:]')"
  case "$var" in
    y|yes|true|1) return 0 ;;
    *) return 1 ;;
  esac
}

# --- venv ---
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

# --- offline root ---
OFFLINE_ROOT=""
OFFLINE_PROJECT_ID=""
OFFLINE_SYSTEM_ID=""
CLUSTER_ID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id) OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id) OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id) CLUSTER_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$OFFLINE_ROOT" ]; then
  offline_prompt_root "${DEFAULT_ROOT}"
else
  OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"
  export PHASE_DATA_ROOT="$OFFLINE_ROOT"
fi

if [ -z "$OFFLINE_PROJECT_ID" ]; then
  offline_select_project
else
  export OFFLINE_PROJECT_ID
fi

if [ -z "$OFFLINE_SYSTEM_ID" ]; then
  offline_select_system
else
  export OFFLINE_SYSTEM_ID
fi

if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi

SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
if [ -n "$CLUSTER_ID" ]; then
  SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid')"
fi
if [ -z "$SAMPLE_LINES" ]; then
  echo "No sampling runs found for this cluster."
  exit 1
fi

SAMPLE_ROW="$(offline_choose_one "Select a sampling run:" "$SAMPLE_LINES")"
SAMPLE_PATH="$(printf "%s" "$SAMPLE_ROW" | awk -F'|' '{print $4}')"
if [ -z "$SAMPLE_PATH" ]; then
  echo "No sampling run selected."
  exit 1
fi

RESULTS_DIR="$SAMPLE_PATH"
if [ -f "$RESULTS_DIR" ]; then
  RESULTS_DIR="$(dirname "$RESULTS_DIR")"
fi

OFFLINE="false"
if prompt_bool "Embed plotly.js for offline viewing? (Y/n)" "Y"; then
  OFFLINE="true"
fi

MODEL_NPZ=""
if prompt_bool "Provide base+delta model paths for Experiment C plots? (y/N)" "N"; then
  MODEL_NPZ="$(prompt "Comma-separated model npz paths (base,delta1,delta2,...)" "")"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_plots
  --results-dir "$RESULTS_DIR"
)

if [ "$OFFLINE" = "true" ]; then
  CMD+=(--offline)
fi
if [ -n "$(trim "$MODEL_NPZ")" ]; then
  CMD+=(--model-npz "$(trim "$MODEL_NPZ")")
fi

echo "Generating HTML plots in: $RESULTS_DIR"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "Done. Open: ${RESULTS_DIR}/index.html"
