#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "${ROOT_DIR}/.venv-potts-fit" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
elif [ -d "${ROOT_DIR}/.venv" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv"
else
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
fi
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_RESULTS="${ROOT_DIR}/results/potts_fit_${TIMESTAMP}"

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
    echo "Create one first: scripts/potts_setup_uv.sh"
  fi
  exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"

NPZ_PATH="$(prompt "Cluster NPZ input path (must exist)" "")"
NPZ_PATH="$(printf "%s" "$NPZ_PATH" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
if [ -z "$NPZ_PATH" ]; then
  echo "Cluster NPZ path is required."
  exit 1
fi
NPZ_DIR="$(dirname "$NPZ_PATH")"
if [ ! -d "$NPZ_DIR" ]; then
  mkdir -p "$NPZ_DIR"
fi
if [ ! -f "$NPZ_PATH" ]; then
  echo "Cluster NPZ not found: $NPZ_PATH"
  exit 1
fi

RESULTS_DIR="$(prompt "Results directory" "${DEFAULT_RESULTS}")"
FIT_METHOD="$(prompt "Fit method (pmi/plm/pmi+plm)" "pmi+plm")"

PLM_DEVICE=""
PLM_EPOCHS=""
PLM_LR=""
PLM_LR_MIN=""
PLM_LR_SCHEDULE=""
PLM_L2=""
PLM_BATCH_SIZE=""
PLM_PROGRESS_EVERY=""
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
fi

CMD=(
  "$PYTHON_BIN" -m phase.simulation.main
  --npz "$NPZ_PATH"
  --results-dir "$RESULTS_DIR"
  --fit-only
  --fit "$FIT_METHOD"
)

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
  )
fi

echo "Running Potts fit..."
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "Done. Potts model saved in: ${RESULTS_DIR}/potts_model.npz"
