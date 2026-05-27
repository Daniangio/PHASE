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

while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id) OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id) OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id) CLUSTER_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  VENV_DIR="${VIRTUAL_ENV}"
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
if [ -z "$OFFLINE_PROJECT_ID" ]; then offline_select_project; fi
if [ -z "$OFFLINE_SYSTEM_ID" ]; then offline_select_system; fi
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
MODEL_ROW="$(offline_choose_one "Select Potts model:" "$MODEL_LINES")"
MODEL_ID="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $1}')"

SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid')"
if [ -z "$(trim "$SAMPLE_LINES")" ]; then
  echo "No samples found in this cluster."
  exit 1
fi

SAMPLE_ROWS="$(offline_choose_multi "Select sample(s) to map onto MD:" "$SAMPLE_LINES")"
if [ -z "$(trim "$SAMPLE_ROWS")" ]; then
  echo "No samples selected."
  exit 1
fi
SAMPLE_IDS=()
while IFS= read -r row; do
  [ -z "$(trim "$row")" ] && continue
  sid="$(sample_id_from_row "$row" || true)"
  if [ -n "$sid" ]; then
    SAMPLE_IDS+=("$sid")
  fi
done <<< "$SAMPLE_ROWS"
if [ "${#SAMPLE_IDS[@]}" -eq 0 ]; then
  echo "Failed to resolve selected sample ids."
  exit 1
fi

MD_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' '$3=="md_eval"')"
if [ -z "$(trim "$MD_LINES")" ]; then
  echo "No md_eval samples found in this cluster."
  exit 1
fi
MD_ROW="$(offline_choose_one "Select MD sample for nearest-neighbor mapping:" "$MD_LINES")"
MD_SAMPLE_ID="$(sample_id_from_row "$MD_ROW" || true)"
if [ -z "$MD_SAMPLE_ID" ]; then
  echo "Failed to resolve MD sample id."
  exit 1
fi

MD_LABEL_MODE="$(prompt "MD label mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MD_LABEL_MODE" != "halo" ]; then MD_LABEL_MODE="assigned"; fi
KEEP_INVALID="$(prompt "Keep invalid rows? (y/N)" "N")"
KEEP_INVALID="$(printf "%s" "$KEEP_INVALID" | tr '[:upper:]' '[:lower:]')"
USE_UNIQUE="$(prompt "Use unique compression? (Y/n)" "Y")"
USE_UNIQUE="$(printf "%s" "$USE_UNIQUE" | tr '[:upper:]' '[:lower:]')"
NORMALIZE="$(prompt "Normalize distances to [0,1]? (Y/n)" "Y")"
NORMALIZE="$(printf "%s" "$NORMALIZE" | tr '[:upper:]' '[:lower:]')"
COMPUTE_PER_RES="$(prompt "Compute per-residue outputs? (Y/n)" "Y")"
COMPUTE_PER_RES="$(printf "%s" "$COMPUTE_PER_RES" | tr '[:upper:]' '[:lower:]')"
ALPHA="$(prompt "Default per-residue alpha (0..1)" "0.75")"
BETA_NODE="$(prompt "Node beta" "1.0")"
BETA_EDGE="$(prompt "Edge beta" "1.0")"
TOP_K="$(prompt "Top-K node-only prefilter candidates (0=disabled)" "0")"
CHUNK_SIZE="$(prompt "Chunk size" "256")"
THRESHOLDS="$(prompt "Distance thresholds (comma separated)" "0.05,0.1,0.2")"
WORKERS="$(prompt "Workers (0=all unique rows)" "0")"
SHOW_PROGRESS="$(prompt "Show progress bars? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_nn_mapping
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --model-id "$MODEL_ID"
  --md-sample-id "$MD_SAMPLE_ID"
  --md-label-mode "$MD_LABEL_MODE"
  --alpha "$ALPHA"
  --beta-node "$BETA_NODE"
  --beta-edge "$BETA_EDGE"
  --chunk-size "$CHUNK_SIZE"
)

for sid in "${SAMPLE_IDS[@]}"; do
  CMD+=(--sample-id "$sid")
done

if [ "$KEEP_INVALID" = "y" ] || [ "$KEEP_INVALID" = "yes" ]; then CMD+=(--keep-invalid); fi
if [ "$USE_UNIQUE" = "n" ] || [ "$USE_UNIQUE" = "no" ]; then CMD+=(--no-unique); fi
if [ "$NORMALIZE" = "n" ] || [ "$NORMALIZE" = "no" ]; then CMD+=(--no-normalize); fi
if [ "$COMPUTE_PER_RES" = "n" ] || [ "$COMPUTE_PER_RES" = "no" ]; then CMD+=(--no-per-residue); fi
if [ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ]; then CMD+=(--progress); fi
if [ -n "$(trim "$TOP_K")" ] && [ "$TOP_K" != "0" ]; then CMD+=(--top-k-candidates "$TOP_K"); fi
if [ -n "$(trim "$WORKERS")" ] && [ "$WORKERS" != "0" ]; then CMD+=(--workers "$WORKERS"); fi
IFS=',' read -r -a THR_ARR <<< "$THRESHOLDS"
for thr in "${THR_ARR[@]}"; do
  thr="$(trim "$thr")"
  [ -z "$thr" ] && continue
  CMD+=(--distance-threshold "$thr")
done

echo ""
printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
