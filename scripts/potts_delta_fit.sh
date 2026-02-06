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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"

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

trim() {
  printf "%s" "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

OFFLINE_ROOT=""
OFFLINE_PROJECT_ID=""
OFFLINE_SYSTEM_ID=""
CLUSTER_ID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root)
      OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id)
      OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id)
      OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id)
      CLUSTER_ID="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

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

if [ -z "$OFFLINE_ROOT" ]; then
  offline_prompt_root "${DEFAULT_ROOT}"
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
MODEL_ROW="$(offline_select_model)"
BASE_MODEL="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $3}')"
if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ID="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $4}')"
fi
if [ -z "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL" ]; then
  echo "Base model not found: $BASE_MODEL"
  exit 1
fi

RESUME_MODEL=""
MODEL_LINES="$(python -m phase.scripts.offline_browser --root "$OFFLINE_ROOT" list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
if [ -n "$MODEL_LINES" ] && prompt_bool "Resume from existing delta model? (y/N)" "N"; then
  RESUME_ROW="$(offline_choose_one "Available Potts models:" "$MODEL_LINES")"
  RESUME_MODEL="$(printf "%s" "$RESUME_ROW" | awk -F'|' '{print $3}')"
  if [ -z "$RESUME_MODEL" ]; then
    echo "No resume model selected."
    exit 1
  fi
fi

MODEL_NAME="$(prompt "Model name (base for delta models)" "")"

NPZ_PATH=""
STATE_IDS=""

CLUSTER_ROW="$(_offline_list list-clusters --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid {print; exit}')"
if [ -z "$CLUSTER_ROW" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi
NPZ_PATH="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $3}')"
if [ -z "$NPZ_PATH" ] || [ ! -f "$NPZ_PATH" ]; then
  echo "Cluster NPZ file is required."
  exit 1
fi

STATE_ROWS="$(offline_select_analysis_states)"
STATE_IDS="$(printf "%s\n" "$STATE_ROWS" | awk -F'|' '{print $1}' | paste -sd, -)"
if [ -z "$STATE_IDS" ]; then
  echo "Select at least one state ID."
  exit 1
fi

EPOCHS="$(prompt "Epochs" "200")"
LR="$(prompt "Learning rate" "1e-3")"
LR_MIN="$(prompt "Learning rate min" "1e-5")"
LR_SCHEDULE="$(prompt "Learning rate schedule (cosine/none)" "cosine")"
BATCH_SIZE="$(prompt "Batch size" "1024")"
SEED="$(prompt "Random seed" "0")"
DEVICE="$(prompt "Device (auto/cuda/cpu)" "auto")"
DEVICE="$(printf "%s" "$DEVICE" | tr '[:upper:]' '[:lower:]')"

DELTA_L2="$(prompt "Delta L2 weight" "1e-5")"
DELTA_GROUP_H="$(prompt "Delta group sparsity (fields)" "1e-3")"
DELTA_GROUP_J="$(prompt "Delta group sparsity (couplings)" "1e-3")"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_delta_fit
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --base-model "$BASE_MODEL"
  --epochs "$EPOCHS"
  --lr "$LR"
  --lr-min "$LR_MIN"
  --lr-schedule "$LR_SCHEDULE"
  --batch-size "$BATCH_SIZE"
  --seed "$SEED"
  --delta-l2 "$DELTA_L2"
  --delta-group-h "$DELTA_GROUP_H"
  --delta-group-j "$DELTA_GROUP_J"
)

if [ -n "$MODEL_NAME" ]; then
  CMD+=(--model-name "$MODEL_NAME")
fi

if [ -n "$RESUME_MODEL" ]; then
  CMD+=(--resume-model "$RESUME_MODEL")
fi

if [ "$DEVICE" != "auto" ] && [ -n "$DEVICE" ]; then
  CMD+=(--device "$DEVICE")
fi

CMD+=(--npz "$NPZ_PATH" --state-ids "$STATE_IDS")

if prompt_bool "Skip saving combined models? (y/N)" "N"; then
  CMD+=(--no-combined)
fi

echo "Running delta Potts fit..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"
echo "Done."
