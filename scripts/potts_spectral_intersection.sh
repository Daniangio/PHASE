#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"

DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"

prompt() {
  local label="$1"
  local default="$2"
  local var
  read -r -p "${label} [${default}]: " var
  if [ -z "$var" ]; then echo "$default"; else echo "$var"; fi
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
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  echo "No active virtual environment detected." >&2
  echo "Activate .venv-phase or .venv-potts-fit first." >&2
  exit 1
fi

if [ -z "$OFFLINE_ROOT" ]; then offline_prompt_root "$DEFAULT_ROOT"; else OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"; export PHASE_DATA_ROOT="$OFFLINE_ROOT"; fi
if [ -z "$OFFLINE_PROJECT_ID" ]; then offline_select_project; fi
if [ -z "$OFFLINE_SYSTEM_ID" ]; then offline_select_system; fi
if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi
[ -z "$CLUSTER_ID" ] && echo "No cluster selected." >&2 && exit 1

SINGLE_LINES="$(_offline_list list-analyses --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" --cluster-id "$CLUSTER_ID" --analysis-type hamiltonian_spectral_single || true)"
[ -z "$(trim "$SINGLE_LINES")" ] && echo "No single-state spectral analyses found. Run Hamiltonian spectral analysis first." >&2 && exit 1
SINGLE_ROW="$(offline_choose_one "Select structural single-state spectral analysis:" "$SINGLE_LINES")"
SINGLE_ID="$(printf "%s" "$SINGLE_ROW" | awk -F'|' '{print $1}')"

PAIR_LINES="$(_offline_list list-analyses --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" --cluster-id "$CLUSTER_ID" --analysis-type hamiltonian_spectral_pair || true)"
[ -z "$(trim "$PAIR_LINES")" ] && echo "No pair spectral analyses found. Run Hamiltonian spectral analysis on at least two states first." >&2 && exit 1
PAIR_ROW="$(offline_choose_one "Select functional pair spectral analysis:" "$PAIR_LINES")"
PAIR_ID="$(printf "%s" "$PAIR_ROW" | awk -F'|' '{print $1}')"

MIN_GROUP_SIZE="$(prompt "Minimum piston group size" "3")"
OVERWRITE="false"
if prompt_bool "Overwrite existing intersection analysis? (y/N)" "N"; then OVERWRITE="true"; fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_spectral_intersection
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --single-analysis-id "$SINGLE_ID"
  --pair-analysis-id "$PAIR_ID"
  --min-group-size "$MIN_GROUP_SIZE"
)
[ "$OVERWRITE" = "true" ] && CMD+=(--overwrite)

echo ""
printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
