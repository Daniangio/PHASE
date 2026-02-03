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

OFFLINE_ROOT=""
OFFLINE_PROJECT_ID=""
OFFLINE_SYSTEM_ID=""
CLUSTER_ID=""
NPZ_PATH=""

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
    --npz)
      NPZ_PATH="$2"; shift 2 ;;
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

if [ -z "$NPZ_PATH" ] || [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
  NPZ_PATH="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $3}')"
fi

if [ -z "$NPZ_PATH" ] || [ ! -f "$NPZ_PATH" ]; then
  echo "Cluster NPZ not found: $NPZ_PATH"
  exit 1
fi

SYSTEM_DIR="${OFFLINE_ROOT}/projects/${OFFLINE_PROJECT_ID}/systems/${OFFLINE_SYSTEM_ID}"
RESULTS_DIR="${SYSTEM_DIR}/tmp/potts_fit_${TIMESTAMP}"

MODEL_NAME=""

RESUME_EXISTING="false"
SELECTED_MODEL_PATH=""
SELECTED_MODEL_NAME=""
RESUME_IN_PLACE="false"
MODEL_LINES="$(python -m phase.scripts.offline_browser --root "$OFFLINE_ROOT" list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
if prompt_bool "Resume existing model fit? (y/N)" "N"; then
  if [ -z "$MODEL_LINES" ]; then
    echo "No existing models found for this cluster."
    exit 1
  fi
  MODEL_ROW="$(offline_choose_one "Available Potts models:" "$MODEL_LINES")"
  SELECTED_MODEL_PATH="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $3}')"
  SELECTED_MODEL_NAME="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $2}')"
  if [ -z "$SELECTED_MODEL_PATH" ]; then
    echo "No model selected."
    exit 1
  fi
  RESUME_EXISTING="true"
fi

FIT_METHOD="plm"
if [ "$RESUME_EXISTING" = "false" ]; then
  MODEL_NAME="$(prompt "Model name" "")"
fi

CONTACT_ALL="false"
CONTACT_PDBS=""
CONTACT_MODE="CA"
CONTACT_CUTOFF="10.0"
if [ "$RESUME_EXISTING" = "false" ]; then
  if prompt_bool "Use all-vs-all edges? (y/N)" "N"; then
    CONTACT_ALL="true"
  else
    PDB_ROWS="$(offline_select_pdbs)"
    if [ -z "$PDB_ROWS" ]; then
      echo "Select at least one PDB unless using all-vs-all."
      exit 1
    fi
    CONTACT_PDBS="$(printf "%s\n" "$PDB_ROWS" | awk -F'|' '{print $3}' | paste -sd, -)"
    CONTACT_MODE="$(prompt "Contact mode (CA/CM)" "CA")"
    CONTACT_MODE="$(printf "%s" "$CONTACT_MODE" | tr '[:lower:]' '[:upper:]')"
    CONTACT_CUTOFF="$(prompt "Contact cutoff (A)" "10.0")"
  fi
else
  echo "Using contact edges from existing model: ${SELECTED_MODEL_NAME:-$SELECTED_MODEL_PATH}"
fi

PLM_DEVICE=""
PLM_EPOCHS=""
PLM_LR=""
PLM_LR_MIN=""
PLM_LR_SCHEDULE=""
PLM_L2=""
PLM_BATCH_SIZE=""
PLM_PROGRESS_EVERY=""
PLM_INIT="pmi"
PLM_INIT_MODEL=""
PLM_RESUME_MODEL=""
PLM_VAL_FRAC="0"
if [ "$FIT_METHOD" != "pmi" ]; then
  PLM_DEVICE="$(prompt "PLM device (auto/cuda/cpu)" "auto")"
  PLM_DEVICE="$(printf "%s" "$PLM_DEVICE" | tr '[:upper:]' '[:lower:]')"
  if [ "$PLM_DEVICE" = "auto" ]; then
    PLM_DEVICE=""
  fi
  PLM_EPOCHS="$(prompt "PLM epochs" "200")"
  PLM_LR="$(prompt "PLM lr" "1e-2")"
  PLM_LR_MIN="$(prompt "PLM lr min" "1e-3")"
  PLM_LR_SCHEDULE="$(prompt "PLM lr schedule (cosine/none)" "cosine")"
  PLM_L2="$(prompt "PLM L2" "1e-5")"
  PLM_BATCH_SIZE="$(prompt "PLM batch size" "512")"
  PLM_PROGRESS_EVERY="$(prompt "PLM progress every" "10")"
  if [ "$RESUME_EXISTING" = "true" ]; then
    PLM_INIT="model"
    PLM_RESUME_MODEL="$SELECTED_MODEL_PATH"
    RESUME_IN_PLACE="true"
  else
    PLM_INIT="$(prompt "PLM init (pmi/zero/model)" "pmi")"
    if [ "$PLM_INIT" = "model" ]; then
      PLM_INIT_MODEL="$(prompt "PLM init model path" "")"
    fi
  fi
  PLM_VAL_FRAC="$(prompt "PLM validation fraction" "0")"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_fit
  --npz "$NPZ_PATH"
  --results-dir "$RESULTS_DIR"
  --fit-only
  --fit "$FIT_METHOD"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
)
if [ -n "$MODEL_NAME" ]; then
  CMD+=(--model-name "$MODEL_NAME")
fi
if [ "$RESUME_IN_PLACE" = "true" ]; then
  CMD+=(--model-out "$SELECTED_MODEL_PATH")
fi

if [ "$RESUME_EXISTING" = "false" ]; then
  if [ "$CONTACT_ALL" = "true" ]; then
    CMD+=(--contact-all-vs-all)
  else
    CMD+=(--pdbs "$CONTACT_PDBS")
  fi
  CMD+=(--contact-atom-mode "$CONTACT_MODE" --contact-cutoff "$CONTACT_CUTOFF")
fi

if [ -n "$PLM_DEVICE" ]; then
  CMD+=(--plm-device "$PLM_DEVICE")
fi

if [ "$FIT_METHOD" != "pmi" ]; then
  CMD+=(
    --plm-epochs "$PLM_EPOCHS"
    --plm-lr "$PLM_LR"
    --plm-lr-min "$PLM_LR_MIN"
    --plm-lr-schedule "$PLM_LR_SCHEDULE"
    --plm-l2 "$PLM_L2"
    --plm-batch-size "$PLM_BATCH_SIZE"
    --plm-progress-every "$PLM_PROGRESS_EVERY"
    --plm-init "$PLM_INIT"
    --plm-val-frac "$PLM_VAL_FRAC"
  )
  if [ -n "$PLM_INIT_MODEL" ]; then
    CMD+=(--plm-init-model "$PLM_INIT_MODEL")
  fi
  if [ -n "$PLM_RESUME_MODEL" ]; then
    CMD+=(--plm-resume-model "$PLM_RESUME_MODEL")
  fi
fi

echo "Running Potts fit..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "Done. Results in: ${RESULTS_DIR}"
echo "Potts model saved under system potts_models (system metadata updated)."
