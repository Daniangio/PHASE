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
DEFAULT_RESULTS="${ROOT_DIR}/results/potts_sampling_${TIMESTAMP}"

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

NPZ_PATH="$(prompt "Cluster NPZ input path (must exist)" "")"
NPZ_PATH="$(trim "$NPZ_PATH")"
if [ -z "$NPZ_PATH" ]; then
  echo "Cluster NPZ path is required."
  exit 1
fi
if [ ! -f "$NPZ_PATH" ]; then
  echo "Cluster NPZ not found: $NPZ_PATH"
  exit 1
fi

MODEL_PATH="$(prompt "Potts model NPZ path (must exist)" "")"
MODEL_PATH="$(trim "$MODEL_PATH")"
if [ -z "$MODEL_PATH" ]; then
  echo "Potts model path is required."
  exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "Potts model not found: $MODEL_PATH"
  exit 1
fi

RESULTS_DIR="$(prompt "Results directory" "${DEFAULT_RESULTS}")"

GIBBS_METHOD="$(prompt "Gibbs method (single/rex)" "rex")"
GIBBS_METHOD="$(printf "%s" "$GIBBS_METHOD" | tr '[:upper:]' '[:lower:]')"

BETA="$(prompt "Target beta (used for report)" "1.0")"

GIBBS_SAMPLES=""
GIBBS_BURNIN=""
GIBBS_THIN=""
REX_BETAS=""
REX_N_REPLICAS=""
REX_BETA_MIN=""
REX_BETA_MAX=""
REX_SPACING=""
REX_ROUNDS=""
REX_BURNIN_ROUNDS=""
REX_SWEEPS_PER_ROUND=""
REX_THIN_ROUNDS=""
REX_CHAINS=""

if [ "$GIBBS_METHOD" = "single" ]; then
  GIBBS_SAMPLES="$(prompt "Gibbs samples" "500")"
  GIBBS_BURNIN="$(prompt "Gibbs burn-in sweeps" "50")"
  GIBBS_THIN="$(prompt "Gibbs thin" "2")"
else
  REX_BETAS="$(prompt "Explicit beta ladder (comma separated, leave blank for auto)" "")"
  if [ -z "$(trim "$REX_BETAS")" ]; then
    REX_BETA_MIN="$(prompt "REX beta min" "0.2")"
    REX_BETA_MAX="$(prompt "REX beta max" "1.0")"
    REX_N_REPLICAS="$(prompt "REX replicas" "8")"
    REX_SPACING="$(prompt "REX spacing (geom/lin)" "geom")"
  fi
  REX_ROUNDS="$(prompt "REX total rounds (split across chains)" "2000")"
  REX_BURNIN_ROUNDS="$(prompt "REX burn-in rounds" "50")"
  REX_SWEEPS_PER_ROUND="$(prompt "REX sweeps per round (Gibbs sweeps per replica)" "2")"
  REX_THIN_ROUNDS="$(prompt "REX thin rounds" "1")"
  REX_CHAINS="$(prompt "REX chain count (runs in parallel, total rounds split)" "1")"
fi

SA_READS="$(prompt "SA reads" "2000")"
SA_SWEEPS="$(prompt "SA sweeps" "2000")"
SA_BETA_HOT="$(prompt "SA beta hot (0 = default)" "0")"
SA_BETA_COLD="$(prompt "SA beta cold (0 = default)" "0")"

ESTIMATE_BETA_EFF="false"
if prompt_bool "Estimate beta_eff? (y/N)" "N"; then
  ESTIMATE_BETA_EFF="true"
  BETA_EFF_GRID="$(prompt "beta_eff grid (comma separated, blank=auto)" "")"
  BETA_EFF_W_MARG="$(prompt "beta_eff weight for marginals" "1.0")"
  BETA_EFF_W_PAIR="$(prompt "beta_eff weight for pairs" "1.0")"
fi

SEED="$(prompt "Random seed" "0")"
SHOW_PROGRESS="false"
if prompt_bool "Show progress bars? (Y/n)" "Y"; then
  SHOW_PROGRESS="true"
fi

CMD=(
  "$PYTHON_BIN" -m phase.simulation.main
  --npz "$NPZ_PATH"
  --model-npz "$MODEL_PATH"
  --results-dir "$RESULTS_DIR"
  --gibbs-method "$GIBBS_METHOD"
  --beta "$BETA"
  --sa-reads "$SA_READS"
  --sa-sweeps "$SA_SWEEPS"
  --seed "$SEED"
)

if [ "$GIBBS_METHOD" = "single" ]; then
  CMD+=(--gibbs-samples "$GIBBS_SAMPLES" --gibbs-burnin "$GIBBS_BURNIN" --gibbs-thin "$GIBBS_THIN")
else
  if [ -n "$(trim "$REX_BETAS")" ]; then
    CMD+=(--rex-betas "$(trim "$REX_BETAS")")
  else
    CMD+=(
      --rex-beta-min "$REX_BETA_MIN"
      --rex-beta-max "$REX_BETA_MAX"
      --rex-n-replicas "$REX_N_REPLICAS"
      --rex-spacing "$REX_SPACING"
    )
  fi
  CMD+=(
    --rex-rounds "$REX_ROUNDS"
    --rex-burnin-rounds "$REX_BURNIN_ROUNDS"
    --rex-sweeps-per-round "$REX_SWEEPS_PER_ROUND"
    --rex-thin-rounds "$REX_THIN_ROUNDS"
    --rex-chain-count "$REX_CHAINS"
  )
fi

if [ "$SA_BETA_HOT" != "0" ] && [ "$SA_BETA_COLD" != "0" ]; then
  CMD+=(--sa-beta-hot "$SA_BETA_HOT" --sa-beta-cold "$SA_BETA_COLD")
fi

if [ "$ESTIMATE_BETA_EFF" = "true" ]; then
  CMD+=(--estimate-beta-eff --beta-eff-w-marg "$BETA_EFF_W_MARG" --beta-eff-w-pair "$BETA_EFF_W_PAIR")
  if [ -n "$(trim "$BETA_EFF_GRID")" ]; then
    CMD+=(--beta-eff-grid "$(trim "$BETA_EFF_GRID")")
  fi
fi

if [ "$SHOW_PROGRESS" = "true" ]; then
  CMD+=(--progress)
fi

echo "Running Potts sampling..."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "Done. Sampling outputs in: ${RESULTS_DIR}"
echo "Summary: ${RESULTS_DIR}/run_summary.npz"
echo "Model copy: ${RESULTS_DIR}/potts_model.npz"
