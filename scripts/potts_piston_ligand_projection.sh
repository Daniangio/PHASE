#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"

DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"
prompt() { local label="$1" default="$2" var; read -r -p "${label} [${default}]: " var; if [ -z "$var" ]; then echo "$default"; else echo "$var"; fi; }

OFFLINE_ROOT=""; OFFLINE_PROJECT_ID=""; OFFLINE_SYSTEM_ID=""; CLUSTER_ID=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) OFFLINE_ROOT="$2"; shift 2 ;;
    --project-id) OFFLINE_PROJECT_ID="$2"; shift 2 ;;
    --system-id) OFFLINE_SYSTEM_ID="$2"; shift 2 ;;
    --cluster-id) CLUSTER_ID="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then PYTHON_BIN="${VIRTUAL_ENV}/bin/python"; else echo "No active virtual environment detected." >&2; exit 1; fi
if [ -z "$OFFLINE_ROOT" ]; then offline_prompt_root "$DEFAULT_ROOT"; else OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"; export PHASE_DATA_ROOT="$OFFLINE_ROOT"; fi
if [ -z "$OFFLINE_PROJECT_ID" ]; then offline_select_project; fi
if [ -z "$OFFLINE_SYSTEM_ID" ]; then offline_select_system; fi
if [ -z "$CLUSTER_ID" ]; then CLUSTER_ROW="$(offline_select_cluster)"; CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"; fi
[ -z "$CLUSTER_ID" ] && echo "No cluster selected." >&2 && exit 1

INTERSECTION_LINES="$(_offline_list list-analyses --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" --cluster-id "$CLUSTER_ID" --analysis-type hamiltonian_spectral_intersection || true)"
[ -z "$(trim "$INTERSECTION_LINES")" ] && echo "No spectral intersection analyses found. Run Spectral set-intersection first." >&2 && exit 1
INTERSECTION_ROW="$(offline_choose_one "Select piston intersection analysis:" "$INTERSECTION_LINES")"
INTERSECTION_ID="$(printf "%s" "$INTERSECTION_ROW" | awk -F'|' '{print $1}')"

SAMPLE_LINES="$(_offline_list list-cluster-samples --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" --cluster-id "$CLUSTER_ID" || true)"
[ -z "$(trim "$SAMPLE_LINES")" ] && echo "No samples found in this cluster." >&2 && exit 1
SAMPLE_ROWS="$(offline_choose_multi "Select ligand/MD samples:" "$SAMPLE_LINES")"
[ -z "$(trim "$SAMPLE_ROWS")" ] && echo "No samples selected." >&2 && exit 1
SAMPLE_IDS="$(printf "%s\n" "$SAMPLE_ROWS" | awk -F'|' 'NF {print $1}' | paste -sd, -)"
[ -z "$SAMPLE_IDS" ] && echo "Could not resolve selected sample ids." >&2 && exit 1

LABEL_MODE="$(prompt "Label mode (assigned/halo)" "assigned")"
OVERWRITE="false"; if prompt_bool "Overwrite existing ligand projection? (y/N)" "N"; then OVERWRITE="true"; fi
CMD=("$PYTHON_BIN" -m phase.scripts.potts_piston_ligand_projection --root "$OFFLINE_ROOT" --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" --cluster-id "$CLUSTER_ID" --intersection-analysis-id "$INTERSECTION_ID" --sample-ids "$SAMPLE_IDS" --label-mode "$LABEL_MODE")
[ "$OVERWRITE" = "true" ] && CMD+=(--overwrite)

echo ""; printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
