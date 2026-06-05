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

STATE_LINES="$(_offline_list list-states --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
[ -z "$(trim "$STATE_LINES")" ] && echo "No states found for this system." >&2 && exit 1
STATE_ROWS="$(offline_choose_multi "Select states for Hamiltonian spectral analysis:" "$STATE_LINES")"
[ -z "$(trim "$STATE_ROWS")" ] && echo "No states selected." >&2 && exit 1
STATE_IDS="$(printf "%s\n" "$STATE_ROWS" | awk -F'|' 'NF {print $1}' | paste -sd, -)"
[ -z "$STATE_IDS" ] && echo "No state ids selected." >&2 && exit 1

TOP_K="$(prompt "Eigenvectors to store" "20")"
OVERWRITE="false"
if prompt_bool "Overwrite existing spectral analyses? (y/N)" "N"; then OVERWRITE="true"; fi
SHOW_PROGRESS="$(prompt "Show progress? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_hamiltonian_spectral
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --state-ids "$STATE_IDS"
  --top-k "$TOP_K"
)
[ "$OVERWRITE" = "true" ] && CMD+=(--overwrite)
[ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ] && CMD+=(--progress)

echo ""
printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
