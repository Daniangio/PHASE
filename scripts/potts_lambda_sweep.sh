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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

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

if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi
if [ -z "$CLUSTER_ID" ]; then
  echo "No cluster selected."
  exit 1
fi

MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
if [ -z "$(trim "$MODEL_LINES")" ]; then
  echo "No Potts models found for cluster: $CLUSTER_ID"
  exit 1
fi

echo ""
echo "Endpoint models:"
echo "  - Model B corresponds to λ=0"
echo "  - Model A corresponds to λ=1"
echo ""
MODEL_B_ROW="$(offline_choose_one "Select endpoint model B (λ=0):" "$MODEL_LINES")"
MODEL_B_ID="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $1}')"
MODEL_B_NAME="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $2}')"
MODEL_A_ROW="$(offline_choose_one "Select endpoint model A (λ=1):" "$MODEL_LINES")"
MODEL_A_ID="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $1}')"
MODEL_A_NAME="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $2}')"
if [ -z "$MODEL_A_ID" ] || [ -z "$MODEL_B_ID" ]; then
  echo "Both endpoint models are required."
  exit 1
fi
if [ "$MODEL_A_ID" = "$MODEL_B_ID" ]; then
  echo "Endpoint models must be different."
  exit 1
fi

SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid')"
MD_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' '$3=="md_eval"')"
if [ -z "$(trim "$MD_LINES")" ]; then
  echo "No MD samples (md_eval) found in this cluster. Run \"Recompute MD samples\" first."
  exit 1
fi
echo ""
echo "Select 3 MD reference samples (these define the comparison targets):"
MD_ROWS="$(offline_choose_multi "Available MD samples:" "$MD_LINES")"
MD_IDS=()

# offline_choose_multi returns full rows; list-sampling columns are: cluster_id|name|type|path
# We need sample_id, so query via project store in python after selecting by name+path is messy.
# Instead, use the sample directory name from the 'path' column (clusters/.../samples/<id>/sample.npz).
while IFS= read -r row; do
  [ -z "$row" ] && continue
  p="$(printf "%s" "$row" | awk -F'|' '{print $4}')"
  p="$(trim "$p")"
  if [ -z "$p" ]; then
    continue
  fi
  base="$(basename "$(dirname "$p")")"
  MD_IDS+=("$base")
done <<< "$MD_ROWS"

if [ "${#MD_IDS[@]}" -ne 3 ]; then
  echo "Please select exactly 3 MD samples (selected: ${#MD_IDS[@]})."
  exit 1
fi

MD_LABEL_MODE="$(prompt "MD label mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MD_LABEL_MODE" != "halo" ]; then
  MD_LABEL_MODE="assigned"
fi

LAMBDA_COUNT="$(prompt "Lambda count (>=2)" "21")"
ALPHA="$(prompt "Alpha for match curve (0..1)" "0.5")"
SERIES_LABEL_DEFAULT="Lambda sweep ${MODEL_B_NAME:-$MODEL_B_ID} -> ${MODEL_A_NAME:-$MODEL_A_ID} ${TIMESTAMP}"
SERIES_LABEL="$(prompt "Series label" "$SERIES_LABEL_DEFAULT")"

GIBBS_METHOD="$(prompt "Gibbs method (single/rex)" "rex")"
GIBBS_METHOD="$(printf "%s" "$GIBBS_METHOD" | tr '[:upper:]' '[:lower:]')"
if [ "$GIBBS_METHOD" != "single" ]; then
  GIBBS_METHOD="rex"
fi

SHOW_PROGRESS="$(prompt "Show progress bars? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"
if [ "$SHOW_PROGRESS" = "n" ] || [ "$SHOW_PROGRESS" = "no" ]; then
  SHOW_PROGRESS="false"
else
  SHOW_PROGRESS="true"
fi

BETA="$(prompt "Target beta" "1.0")"
SEED="$(prompt "Random seed" "0")"

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

if [ "$GIBBS_METHOD" = "single" ]; then
  GIBBS_SAMPLES="$(prompt "Gibbs samples" "500")"
  GIBBS_BURNIN="$(prompt "Gibbs burn-in sweeps" "50")"
  GIBBS_THIN="$(prompt "Gibbs thin" "2")"
  GIBBS_CHAINS="$(prompt "Gibbs chain count (parallel independent chains)" "1")"
else
  REX_BETAS="$(prompt "Explicit beta ladder (comma separated, leave blank for auto)" "")"
  if [ -z "$(trim "$REX_BETAS")" ]; then
    REX_BETA_MIN="$(prompt "REX beta min" "0.2")"
    REX_BETA_MAX="$(prompt "REX beta max" "1.0")"
    REX_N_REPLICAS="$(prompt "REX replicas" "8")"
    REX_SPACING="$(prompt "REX spacing (geom/lin)" "geom")"
  fi
  REX_ROUNDS="$(prompt "REX total rounds" "2000")"
  REX_BURNIN_ROUNDS="$(prompt "REX burn-in rounds" "50")"
  REX_SWEEPS_PER_ROUND="$(prompt "REX sweeps per round" "2")"
  REX_THIN_ROUNDS="$(prompt "REX thin rounds" "1")"
  REX_CHAINS="$(prompt "REX chain count (runs in parallel, total rounds split)" "1")"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_lambda_sweep
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --model-a-id "$MODEL_A_ID"
  --model-b-id "$MODEL_B_ID"
  --md-sample-id-1 "${MD_IDS[0]}"
  --md-sample-id-2 "${MD_IDS[1]}"
  --md-sample-id-3 "${MD_IDS[2]}"
  --md-label-mode "$MD_LABEL_MODE"
  --lambda-count "$LAMBDA_COUNT"
  --alpha "$ALPHA"
  --series-label "$SERIES_LABEL"
  --gibbs-method "$GIBBS_METHOD"
  --beta "$BETA"
  --seed "$SEED"
)

if [ "$SHOW_PROGRESS" = "true" ]; then
  CMD+=(--progress)
fi

if [ "$GIBBS_METHOD" = "single" ]; then
  CMD+=(
    --gibbs-samples "$GIBBS_SAMPLES"
    --gibbs-burnin "$GIBBS_BURNIN"
    --gibbs-thin "$GIBBS_THIN"
    --gibbs-chains "$GIBBS_CHAINS"
  )
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
    --rex-chains "$REX_CHAINS"
  )
fi

echo ""
echo "Running lambda sweep..."
echo "  cluster: $CLUSTER_ID"
echo "  model B (λ=0): ${MODEL_B_NAME:-$MODEL_B_ID} (${MODEL_B_ID})"
echo "  model A (λ=1): ${MODEL_A_NAME:-$MODEL_A_ID} (${MODEL_A_ID})"
echo "  MD refs: ${MD_IDS[*]}"
echo ""

exec "${CMD[@]}"
