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
  echo "Using active virtual environment at: ${VIRTUAL_ENV}"
else
  echo "No active virtual environment detected." >&2
  echo "Activate .venv-potts-fit or .venv first." >&2
  exit 1
fi

if [ -z "$OFFLINE_ROOT" ]; then
  offline_prompt_root "$DEFAULT_ROOT"
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
  echo "No cluster selected." >&2
  exit 1
fi

USE_MODELS="false"
if prompt_bool "Use Potts model pair (for edge set + optional ref inference)? (y/N)" "N"; then
  USE_MODELS="true"
fi

MODEL_A_ID=""
MODEL_B_ID=""
if [ "$USE_MODELS" = "true" ]; then
  MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
  MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
  if [ -z "$(trim "$MODEL_LINES")" ]; then
    echo "No Potts models found for cluster: $CLUSTER_ID" >&2
    exit 1
  fi
  MODEL_A_ROW="$(offline_choose_one "Select model A:" "$MODEL_LINES")"
  MODEL_A_ID="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $1}')"
  MODEL_B_ROW="$(offline_choose_one "Select model B:" "$MODEL_LINES")"
  MODEL_B_ID="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $1}')"
  if [ -z "$MODEL_A_ID" ] || [ -z "$MODEL_B_ID" ]; then
    echo "Model A and B are required." >&2
    exit 1
  fi
  if [ "$MODEL_A_ID" = "$MODEL_B_ID" ]; then
    echo "Model A and B must be different." >&2
    exit 1
  fi
fi

export _PHASE_DJS_ROOT="$OFFLINE_ROOT"
export _PHASE_DJS_PROJECT="$OFFLINE_PROJECT_ID"
export _PHASE_DJS_SYSTEM="$OFFLINE_SYSTEM_ID"
export _PHASE_DJS_CLUSTER="$CLUSTER_ID"
SAMPLE_LINES="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from phase.services.project_store import ProjectStore

root = Path(os.environ["_PHASE_DJS_ROOT"]) / "projects"
store = ProjectStore(base_dir=root)
samples = store.list_samples(os.environ["_PHASE_DJS_PROJECT"], os.environ["_PHASE_DJS_SYSTEM"], os.environ["_PHASE_DJS_CLUSTER"])
for s in samples:
    sid = str(s.get("sample_id") or "").strip()
    if not sid:
        continue
    name = str(s.get("name") or sid)
    typ = str(s.get("type") or "")
    state = str(s.get("state_id") or "")
    print(f"{sid}|{name}|{typ}|{state}")
PY
)"
unset _PHASE_DJS_ROOT _PHASE_DJS_PROJECT _PHASE_DJS_SYSTEM _PHASE_DJS_CLUSTER
if [ -z "$(trim "$SAMPLE_LINES")" ]; then
  echo "No samples found for this cluster." >&2
  exit 1
fi
SAMPLE_ROWS="$(offline_choose_multi "Select samples to compute JS analysis on:" "$SAMPLE_LINES")"
SAMPLE_IDS="$(printf "%s\n" "$SAMPLE_ROWS" | awk -F'|' '{print $1}' | awk 'NF' | paste -sd',' -)"
if [ -z "$SAMPLE_IDS" ]; then
  echo "No samples selected." >&2
  exit 1
fi

REF_A_IDS=""
REF_B_IDS=""
if [ "$USE_MODELS" = "true" ] && prompt_bool "Infer references from model state_ids? (Y/n)" "Y"; then
  :
else
  MD_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' '$3=="md_eval"')"
  if [ -z "$(trim "$MD_LINES")" ]; then
    echo "No md_eval samples available for manual references." >&2
    exit 1
  fi
  REF_A_ROWS="$(offline_choose_multi "Select reference samples for side A:" "$MD_LINES")"
  REF_B_ROWS="$(offline_choose_multi "Select reference samples for side B:" "$MD_LINES")"
  REF_A_IDS="$(printf "%s\n" "$REF_A_ROWS" | awk -F'|' '{print $1}' | awk 'NF' | paste -sd',' -)"
  REF_B_IDS="$(printf "%s\n" "$REF_B_ROWS" | awk -F'|' '{print $1}' | awk 'NF' | paste -sd',' -)"
  if [ -z "$REF_A_IDS" ] || [ -z "$REF_B_IDS" ]; then
    echo "Reference samples for both A and B are required." >&2
    exit 1
  fi
fi

EDGE_MODE=""
CONTACT_STATE_IDS=""
CONTACT_PDBS=""
CONTACT_CUTOFF="10.0"
CONTACT_ATOM_MODE="CA"
if [ "$USE_MODELS" != "true" ]; then
  EDGE_MODE="$(prompt "Edge mode (cluster/all_vs_all/contact)" "contact")"
  EDGE_MODE="$(printf "%s" "$EDGE_MODE" | tr '[:upper:]' '[:lower:]')"
  case "$EDGE_MODE" in
    cluster|all_vs_all|contact) ;;
    *) EDGE_MODE="contact" ;;
  esac
  if [ "$EDGE_MODE" = "contact" ]; then
    if prompt_bool "Select contact PDBs from states? (Y/n)" "Y"; then
      STATE_ROWS="$(offline_select_states)"
      CONTACT_STATE_IDS="$(printf "%s\n" "$STATE_ROWS" | awk -F'|' '{print $1}' | awk 'NF' | paste -sd',' -)"
    fi
    CONTACT_PDBS="$(prompt "Extra contact PDB paths (comma separated, optional)" "")"
    CONTACT_PDBS="$(trim "$CONTACT_PDBS")"
    CONTACT_CUTOFF="$(prompt "Contact cutoff (A)" "10.0")"
    CONTACT_ATOM_MODE="$(prompt "Contact atom mode (CA/CM)" "CA")"
    CONTACT_ATOM_MODE="$(printf "%s" "$CONTACT_ATOM_MODE" | tr '[:lower:]' '[:upper:]')"
    if [ "$CONTACT_ATOM_MODE" != "CM" ]; then CONTACT_ATOM_MODE="CA"; fi
    if [ -z "$CONTACT_STATE_IDS" ] && [ -z "$CONTACT_PDBS" ]; then
      echo "edge_mode=contact requires at least one state or PDB path." >&2
      exit 1
    fi
  fi
fi

MD_LABEL_MODE="$(prompt "MD labels mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
if [ "$MD_LABEL_MODE" != "halo" ]; then MD_LABEL_MODE="assigned"; fi
KEEP_INVALID="false"
if prompt_bool "Keep invalid frames? (y/N)" "N"; then KEEP_INVALID="true"; fi
TOP_K_RES="$(prompt "Top residues" "20")"
TOP_K_EDGES="$(prompt "Top edges" "2000")"
ALPHA="$(prompt "Node/edge mix alpha [0..1]" "0.5")"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_delta_js
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --sample-ids "$SAMPLE_IDS"
  --md-label-mode "$MD_LABEL_MODE"
  --top-k-residues "$TOP_K_RES"
  --top-k-edges "$TOP_K_EDGES"
  --node-edge-alpha "$ALPHA"
)

if [ "$USE_MODELS" = "true" ]; then
  CMD+=(--model-a-id "$MODEL_A_ID" --model-b-id "$MODEL_B_ID")
else
  CMD+=(--edge-mode "$EDGE_MODE")
  if [ -n "$CONTACT_STATE_IDS" ]; then CMD+=(--contact-state-ids "$CONTACT_STATE_IDS"); fi
  if [ -n "$CONTACT_PDBS" ]; then CMD+=(--contact-pdbs "$CONTACT_PDBS"); fi
  if [ "$EDGE_MODE" = "contact" ]; then
    CMD+=(--contact-cutoff "$CONTACT_CUTOFF" --contact-atom-mode "$CONTACT_ATOM_MODE")
  fi
fi
if [ -n "$REF_A_IDS" ]; then CMD+=(--ref-a-sample-ids "$REF_A_IDS"); fi
if [ -n "$REF_B_IDS" ]; then CMD+=(--ref-b-sample-ids "$REF_B_IDS"); fi
if [ "$KEEP_INVALID" = "true" ]; then CMD+=(--keep-invalid); fi

echo ""
echo "Running delta JS analysis..."
echo "  cluster: $CLUSTER_ID"
if [ "$USE_MODELS" = "true" ]; then
  echo "  model A: $MODEL_A_ID"
  echo "  model B: $MODEL_B_ID"
else
  echo "  model pair: (none)"
  echo "  edge mode: $EDGE_MODE"
fi
echo "  sample_ids: $SAMPLE_IDS"
echo ""
exec "${CMD[@]}"
