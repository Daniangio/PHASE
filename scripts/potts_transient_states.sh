#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"

DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"
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
  echo "Activate .venv-phase or another PHASE environment first." >&2
  exit 1
fi

if [ -z "$OFFLINE_ROOT" ]; then offline_prompt_root "$DEFAULT_ROOT"; else OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"; export PHASE_DATA_ROOT="$OFFLINE_ROOT"; fi
if [ -z "$OFFLINE_PROJECT_ID" ]; then offline_select_project; fi
if [ -z "$OFFLINE_SYSTEM_ID" ]; then offline_select_system; fi
if [ -z "$CLUSTER_ID" ]; then
  CLUSTER_ROW="$(offline_select_cluster)"
  CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
fi
if [ -z "$CLUSTER_ID" ]; then echo "No cluster selected." >&2; exit 1; fi

export _PHASE_TS_ROOT="$OFFLINE_ROOT"
export _PHASE_TS_PROJECT="$OFFLINE_PROJECT_ID"
export _PHASE_TS_SYSTEM="$OFFLINE_SYSTEM_ID"
export _PHASE_TS_CLUSTER="$CLUSTER_ID"
SAMPLE_LINES="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from phase.services.project_store import ProjectStore
root = Path(os.environ['_PHASE_TS_ROOT']) / 'projects'
store = ProjectStore(base_dir=root)
samples = store.list_samples(os.environ['_PHASE_TS_PROJECT'], os.environ['_PHASE_TS_SYSTEM'], os.environ['_PHASE_TS_CLUSTER'])
for s in samples:
    sid = str(s.get('sample_id') or '').strip()
    if not sid:
        continue
    name = str(s.get('name') or sid)
    typ = str(s.get('type') or '')
    state = str(s.get('state_id') or '')
    print(f"{sid}|{name}|{typ}|{state}")
PY
)"
unset _PHASE_TS_ROOT _PHASE_TS_PROJECT _PHASE_TS_SYSTEM _PHASE_TS_CLUSTER
if [ -z "$(trim "$SAMPLE_LINES")" ]; then echo "No samples found for this cluster." >&2; exit 1; fi
SAMPLE_ROWS="$(offline_choose_multi "Select samples/trajectories to compare:" "$SAMPLE_LINES")"
SAMPLE_IDS="$(printf "%s\n" "$SAMPLE_ROWS" | awk -F'|' '{print $1}' | awk 'NF' | paste -sd',' -)"
if [ -z "$SAMPLE_IDS" ]; then echo "No samples selected." >&2; exit 1; fi

MD_LABEL_MODE="$(prompt "MD labels mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MD_LABEL_MODE" != "halo" ]; then MD_LABEL_MODE="assigned"; fi
KEEP_INVALID="false"
if prompt_bool "Keep invalid frames? (y/N)" "N"; then KEEP_INVALID="true"; fi
P_MIN="$(prompt "Minimum transient occupancy" "0.005")"
P_MAX="$(prompt "Maximum transient occupancy" "0.05")"
ENRICH="$(prompt "Minimum log2 enrichment" "1.0")"
TOP_NODES="$(prompt "Top node hits to store" "500")"
INCLUDE_EDGES="true"
if prompt_bool "Compute edge transient states? (Y/n)" "Y"; then INCLUDE_EDGES="true"; else INCLUDE_EDGES="false"; fi
EDGE_MODE="cluster"
TOP_EDGES="1000"
DELTA_PMI=""
if [ "$INCLUDE_EDGES" = "true" ]; then
  EDGE_MODE="$(prompt "Edge mode (cluster/all_vs_all)" "cluster")"
  EDGE_MODE="$(printf "%s" "$EDGE_MODE" | tr '[:upper:]' '[:lower:]')"
  if [ "$EDGE_MODE" != "all_vs_all" ]; then EDGE_MODE="cluster"; fi
  DELTA_PMI="$(prompt "Minimum ΔPMI for edge hits (blank = no cutoff)" "")"
  TOP_EDGES="$(prompt "Top edge hits to store" "1000")"
fi

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_transient_states
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --sample-ids "$SAMPLE_IDS"
  --md-label-mode "$MD_LABEL_MODE"
  --p-min "$P_MIN"
  --p-max "$P_MAX"
  --enrichment-min "$ENRICH"
  --top-k-nodes "$TOP_NODES"
  --edge-mode "$EDGE_MODE"
  --top-k-edges "$TOP_EDGES"
  --progress
)
if [ "$KEEP_INVALID" = "true" ]; then CMD+=(--keep-invalid); fi
if [ "$INCLUDE_EDGES" != "true" ]; then CMD+=(--no-edges); fi
if [ -n "$(trim "$DELTA_PMI")" ]; then CMD+=(--delta-pmi-min "$DELTA_PMI"); fi

echo ""
echo "Running transient-state analysis..."
echo "  cluster: $CLUSTER_ID"
echo "  sample_ids: $SAMPLE_IDS"
echo ""
exec "${CMD[@]}"
