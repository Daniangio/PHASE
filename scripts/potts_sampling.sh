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
SAMPLE_ID="$(python - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"

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

sample_id_from_row() {
  local row="$1"
  local p
  p="$(printf "%s" "$row" | awk -F'|' '{print $4}')"
  p="$(trim "$p")"
  if [ -z "$p" ]; then
    return 1
  fi
  basename "$(dirname "$p")"
}

OFFLINE_ROOT=""
OFFLINE_PROJECT_ID=""
OFFLINE_SYSTEM_ID=""
CLUSTER_ID=""
NPZ_PATH=""
MODEL_PATHS=()

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
    --model-npz)
      IFS=',' read -r -a _MODEL_PARTS <<< "$2"
      for _part in "${_MODEL_PARTS[@]}"; do
        _part="$(trim "$_part")"
        if [ -n "$_part" ]; then
          MODEL_PATHS+=("$_part")
        fi
      done
      shift 2 ;;
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

if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
  MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
  if [ -n "$CLUSTER_ID" ]; then
    MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
  fi
  MODEL_ROWS="$(offline_choose_multi "Available Potts models:" "$MODEL_LINES")"
  while IFS= read -r row; do
    [ -z "$row" ] && continue
    path="$(printf "%s" "$row" | awk -F'|' '{print $3}')"
    if [ -n "$path" ]; then
      MODEL_PATHS+=("$path")
    fi
  done <<< "$MODEL_ROWS"
fi
if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
  echo "No Potts model selected."
  exit 1
fi
for model_path in "${MODEL_PATHS[@]}"; do
  if [ ! -f "$model_path" ]; then
    echo "Potts model not found: $model_path"
    exit 1
  fi
done

RESULTS_DIR="$(python - <<PY
from pathlib import Path
from phase.services.project_store import ProjectStore
root = Path("${OFFLINE_ROOT}") / "projects"
store = ProjectStore(base_dir=root)
dirs = store.ensure_cluster_directories("${OFFLINE_PROJECT_ID}", "${OFFLINE_SYSTEM_ID}", "${CLUSTER_ID}")
cluster_dir = dirs["cluster_dir"]
out = cluster_dir / "samples" / "${SAMPLE_ID}"
out.mkdir(parents=True, exist_ok=True)
print(out)
PY
)"

SAMPLE_NAME="$(prompt "Sample name" "Sampling ${TIMESTAMP}")"

SAMPLING_METHOD="$(prompt "Sampling method (gibbs/sa)" "gibbs")"
SAMPLING_METHOD="$(printf "%s" "$SAMPLING_METHOD" | tr '[:upper:]' '[:lower:]')"
if [ "$SAMPLING_METHOD" != "sa" ]; then
  SAMPLING_METHOD="gibbs"
fi

GIBBS_METHOD="rex"
if [ "$SAMPLING_METHOD" = "gibbs" ]; then
  GIBBS_METHOD="$(prompt "Gibbs method (single/rex)" "rex")"
  GIBBS_METHOD="$(printf "%s" "$GIBBS_METHOD" | tr '[:upper:]' '[:lower:]')"
fi

BETA="$(prompt "Target beta" "1.0")"

GIBBS_SAMPLES=""
GIBBS_BURNIN=""
GIBBS_THIN=""
GIBBS_CHAINS=""
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

if [ "$SAMPLING_METHOD" = "gibbs" ]; then
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
    REX_ROUNDS="$(prompt "REX total rounds (split across chains)" "2000")"
    REX_BURNIN_ROUNDS="$(prompt "REX burn-in rounds" "50")"
    REX_SWEEPS_PER_ROUND="$(prompt "REX sweeps per round (Gibbs sweeps per replica)" "2")"
    REX_THIN_ROUNDS="$(prompt "REX thin rounds" "1")"
    REX_CHAINS="$(prompt "REX chain count (runs in parallel, total rounds split)" "1")"
  fi
else
  SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
  SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid')"
  MD_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' '$3=="md_eval"')"
  if [ -z "$(trim "$MD_LINES")" ]; then
    echo "No md_eval samples found in this cluster."
    exit 1
  fi
  MD_ROW="$(offline_choose_one "Select MD sample for SA warm-starts:" "$MD_LINES")"
  MD_SAMPLE_ID="$(sample_id_from_row "$MD_ROW" || true)"
  if [ -z "$MD_SAMPLE_ID" ]; then
    echo "Failed to resolve MD sample id."
    exit 1
  fi
  SA_READS="$(prompt "SA reads" "2000")"
  SA_CHAINS="$(prompt "SA chain count (parallel independent chains)" "1")"
  SA_SWEEPS="$(prompt "SA sweeps" "2000")"
  SA_SCHEDULE_MODE="$(prompt "SA schedule mode (auto/range/custom)" "range")"
  SA_SCHEDULE_MODE="$(printf "%s" "$SA_SCHEDULE_MODE" | tr '[:upper:]' '[:lower:]')"
  if [ "$SA_SCHEDULE_MODE" != "range" ] && [ "$SA_SCHEDULE_MODE" != "custom" ]; then
    SA_SCHEDULE_MODE="auto"
  fi
  SA_SCHEDULE_TYPE="geometric"
  SA_CUSTOM_BETA_SCHEDULE=""
  SA_BETA_HOT="0"
  SA_BETA_COLD="0"
  if [ "$SA_SCHEDULE_MODE" = "custom" ]; then
    SA_CUSTOM_BETA_SCHEDULE="$(prompt "SA custom beta schedule (comma separated)" "0.8,1.0,1.5,2.0,3.0,5.0,8.0,10.0")"
  else
    SA_SCHEDULE_TYPE="$(prompt "SA schedule type (geometric/linear)" "geometric")"
    SA_SCHEDULE_TYPE="$(printf "%s" "$SA_SCHEDULE_TYPE" | tr '[:upper:]' '[:lower:]')"
    if [ "$SA_SCHEDULE_TYPE" != "linear" ]; then
      SA_SCHEDULE_TYPE="geometric"
    fi
    if [ "$SA_SCHEDULE_MODE" = "range" ]; then
      SA_BETA_HOT="$(prompt "SA beta hot" "0.01")"
      SA_BETA_COLD="$(prompt "SA beta cold" "2.0")"
    fi
  fi
  SA_NUM_SWEEPS_PER_BETA="$(prompt "SA sweeps per beta" "2")"
  SA_ACCEPTANCE_CRITERIA="$(prompt "SA acceptance criteria (Metropolis/Gibbs)" "Metropolis")"
  if [ "$SA_ACCEPTANCE_CRITERIA" != "Gibbs" ]; then
    SA_ACCEPTANCE_CRITERIA="Metropolis"
  fi
  SA_RANDOMIZE_ORDER="false"
  if prompt_bool "Randomize SA update order? (y/N)" "N"; then
    SA_RANDOMIZE_ORDER="true"
  fi
  SA_RESTART="$(prompt "SA restart (previous/md/independent)" "independent")"
  SA_RESTART="$(printf "%s" "$SA_RESTART" | tr '[:upper:]' '[:lower:]')"
  if [ "$SA_RESTART" != "md" ] && [ "$SA_RESTART" != "independent" ]; then
    SA_RESTART="previous"
  fi
  SA_INIT="$(prompt "SA init (md/md-frame/random-h/random-uniform)" "md")"
  SA_INIT="$(printf "%s" "$SA_INIT" | tr '[:upper:]' '[:lower:]')"
  if [ "$SA_INIT" != "md-frame" ] && [ "$SA_INIT" != "random-h" ] && [ "$SA_INIT" != "random-uniform" ]; then
    SA_INIT="md"
  fi
  SA_INIT_MD_FRAME=""
  if [ "$SA_INIT" = "md-frame" ]; then
    SA_INIT_MD_FRAME="$(prompt "SA init MD frame index" "0")"
  fi
  SA_MD_STATE_IDS="$(prompt "SA MD state IDs (comma separated, blank = all)" "")"
  PENALTY_SAFETY="$(prompt "Penalty safety" "4.0")"
  REPAIR="$(prompt "Decode repair (none/argmax)" "none")"
  if [ "$REPAIR" != "argmax" ]; then
    REPAIR="none"
  fi
fi

SEED="$(prompt "Random seed" "0")"
SHOW_PROGRESS="false"
if prompt_bool "Show progress bars? (Y/n)" "Y"; then
  SHOW_PROGRESS="true"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_sample
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --sample-id "$SAMPLE_ID"
  --sample-name "$SAMPLE_NAME"
  --npz "$NPZ_PATH"
  --results-dir "$RESULTS_DIR"
  --sampling-method "$SAMPLING_METHOD"
  --gibbs-method "$GIBBS_METHOD"
  --beta "$BETA"
  --seed "$SEED"
)
for model_path in "${MODEL_PATHS[@]}"; do
  CMD+=(--model-npz "$model_path")
done

if [ "$SAMPLING_METHOD" = "gibbs" ] && [ "$GIBBS_METHOD" = "single" ]; then
  CMD+=(--gibbs-samples "$GIBBS_SAMPLES" --gibbs-burnin "$GIBBS_BURNIN" --gibbs-thin "$GIBBS_THIN" --gibbs-chains "$GIBBS_CHAINS")
elif [ "$SAMPLING_METHOD" = "gibbs" ]; then
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

if [ "$SAMPLING_METHOD" = "sa" ]; then
  CMD+=(
    --root "$OFFLINE_ROOT"
    --sa-reads "$SA_READS"
    --sa-chains "$SA_CHAINS"
    --sa-sweeps "$SA_SWEEPS"
    --sa-schedule-type "$SA_SCHEDULE_TYPE"
    --sa-num-sweeps-per-beta "$SA_NUM_SWEEPS_PER_BETA"
    --sa-acceptance-criteria "$SA_ACCEPTANCE_CRITERIA"
    --sa-init "$SA_INIT"
    --sa-restart "$SA_RESTART"
    --md-sample-id "$MD_SAMPLE_ID"
    --penalty-safety "$PENALTY_SAFETY"
    --repair "$REPAIR"
  )
  if [ "$SA_RANDOMIZE_ORDER" = "true" ]; then
    CMD+=(--sa-randomize-order)
  else
    CMD+=(--no-sa-randomize-order)
  fi
  if [ -n "$(trim "$SA_INIT_MD_FRAME")" ]; then
    CMD+=(--sa-init-md-frame "$(trim "$SA_INIT_MD_FRAME")")
  fi
  if [ -n "$(trim "$SA_MD_STATE_IDS")" ]; then
    CMD+=(--sa-md-state-ids "$(trim "$SA_MD_STATE_IDS")")
  fi
fi

if [ "$SAMPLING_METHOD" = "sa" ] && [ "$SA_BETA_HOT" != "0" ] && [ "$SA_BETA_COLD" != "0" ]; then
  CMD+=(--sa-beta-hot "$SA_BETA_HOT" --sa-beta-cold "$SA_BETA_COLD")
fi
if [ "$SAMPLING_METHOD" = "sa" ] && { [ "$SA_SCHEDULE_MODE" = "auto" ] || [ "$SA_SCHEDULE_MODE" = "custom" ]; }; then
  CMD+=(--sa-beta-hot 0 --sa-beta-cold 0)
fi
if [ "$SAMPLING_METHOD" = "sa" ] && [ "$SA_SCHEDULE_MODE" = "custom" ] && [ -n "$(trim "$SA_CUSTOM_BETA_SCHEDULE")" ]; then
  CMD+=(--sa-custom-beta-schedule "$(trim "$SA_CUSTOM_BETA_SCHEDULE")" --sa-schedule-type custom)
fi

if [ "$SHOW_PROGRESS" = "true" ]; then
  CMD+=(--progress)
fi

echo "Running Potts sampling..."
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${CMD[@]}"

echo "Done. Sampling outputs in: ${RESULTS_DIR}"
echo "Sample: ${RESULTS_DIR}/sample.npz"
