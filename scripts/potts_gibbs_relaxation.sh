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
MODEL_ROW="$(offline_choose_one "Select target Potts model:" "$MODEL_LINES")"
MODEL_ID="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $1}')"
MODEL_NAME="$(printf "%s" "$MODEL_ROW" | awk -F'|' '{print $2}')"
if [ -z "$MODEL_ID" ]; then
  echo "No model selected."
  exit 1
fi

export _PHASE_GBR_ROOT="$OFFLINE_ROOT"
export _PHASE_GBR_PROJECT="$OFFLINE_PROJECT_ID"
export _PHASE_GBR_SYSTEM="$OFFLINE_SYSTEM_ID"
export _PHASE_GBR_CLUSTER="$CLUSTER_ID"
MD_LINES="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from phase.services.project_store import ProjectStore

root = Path(os.environ["_PHASE_GBR_ROOT"]) / "projects"
store = ProjectStore(base_dir=root)
system = store.get_system(os.environ["_PHASE_GBR_PROJECT"], os.environ["_PHASE_GBR_SYSTEM"])
cluster_id = os.environ["_PHASE_GBR_CLUSTER"]

entry = next((c for c in (system.metastable_clusters or []) if str(c.get("cluster_id")) == cluster_id), None)
if not isinstance(entry, dict):
    raise SystemExit(0)
for sample in (entry.get("samples") or []):
    if str(sample.get("type")) != "md_eval":
        continue
    sid = str(sample.get("sample_id") or "").strip()
    if not sid:
        continue
    name = str(sample.get("name") or sid)
    print(f"{sid}|{name}")
PY
)"
unset _PHASE_GBR_ROOT _PHASE_GBR_PROJECT _PHASE_GBR_SYSTEM _PHASE_GBR_CLUSTER

if [ -z "$(trim "$MD_LINES")" ]; then
  echo "No md_eval samples found for this cluster."
  exit 1
fi

START_ROW="$(offline_choose_one "Select starting MD sample:" "$MD_LINES")"
START_SAMPLE_ID="$(printf "%s" "$START_ROW" | awk -F'|' '{print $1}')"
START_SAMPLE_NAME="$(printf "%s" "$START_ROW" | awk -F'|' '{print $2}')"
if [ -z "$START_SAMPLE_ID" ]; then
  echo "No starting sample selected."
  exit 1
fi

BETA="$(prompt "Target beta" "1.0")"
N_START="$(prompt "Random starting frames" "100")"
N_SWEEPS="$(prompt "Gibbs sweeps per start" "1000")"
WORKERS="$(prompt "Worker processes (0 = all cpus)" "0")"
SEED="$(prompt "Random seed" "0")"
LABEL_MODE="$(prompt "Start label mode (assigned/halo)" "assigned")"
LABEL_MODE="$(printf "%s" "$LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$LABEL_MODE" != "halo" ]; then
  LABEL_MODE="assigned"
fi

KEEP_INVALID="$(prompt "Keep invalid frames? (y/N)" "N")"
KEEP_INVALID="$(printf "%s" "$KEEP_INVALID" | tr '[:upper:]' '[:lower:]')"
SHOW_PROGRESS="$(prompt "Show progress output? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_gibbs_relaxation
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --model-id "$MODEL_ID"
  --start-sample-id "$START_SAMPLE_ID"
  --beta "$BETA"
  --n-start-frames "$N_START"
  --gibbs-sweeps "$N_SWEEPS"
  --workers "$WORKERS"
  --seed "$SEED"
  --start-label-mode "$LABEL_MODE"
)

if [ "$KEEP_INVALID" = "y" ] || [ "$KEEP_INVALID" = "yes" ] || [ "$KEEP_INVALID" = "true" ]; then
  CMD+=(--keep-invalid)
fi
if [ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ]; then
  CMD+=(--progress)
fi

echo ""
echo "Running Gibbs relaxation analysis..."
echo "  cluster: $CLUSTER_ID"
echo "  model: ${MODEL_NAME:-$MODEL_ID} ($MODEL_ID)"
echo "  start sample: ${START_SAMPLE_NAME:-$START_SAMPLE_ID} ($START_SAMPLE_ID)"
echo ""

exec "${CMD[@]}"

