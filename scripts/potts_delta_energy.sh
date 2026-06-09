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

sample_id_from_row() {
  local row="$1"
  local p
  p="$(printf "%s" "$row" | awk -F'|' '{print $4}')"
  p="$(trim "$p")"
  [ -z "$p" ] && return 1
  basename "$(dirname "$p")"
}

sample_frames_from_row() {
  local row="$1"
  local path
  path="$(printf "%s" "$row" | awk -F'|' '{print $4}')"
  path="$(trim "$path")"
  [ -z "$path" ] && echo "0" && return 0
  "$PYTHON_BIN" - <<PY
import numpy as np
p = r"$path"
try:
    with np.load(p, allow_pickle=False) as data:
        for key in ("labels", "labels_assigned", "samples"):
            if key in data:
                print(int(np.asarray(data[key]).shape[0])); break
        else:
            print(0)
except Exception:
    print(0)
PY
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

MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$4==cid')"
[ -z "$(trim "$MODEL_LINES")" ] && echo "No Potts models found for cluster: $CLUSTER_ID" >&2 && exit 1
MODEL_A_ROW="$(offline_choose_one "Select model A:" "$MODEL_LINES")"
MODEL_A_ID="$(printf "%s" "$MODEL_A_ROW" | awk -F'|' '{print $1}')"
MODEL_B_ROW="$(offline_choose_one "Select model B:" "$MODEL_LINES")"
MODEL_B_ID="$(printf "%s" "$MODEL_B_ROW" | awk -F'|' '{print $1}')"
[ -z "$MODEL_A_ID" ] || [ -z "$MODEL_B_ID" ] && echo "Model A and model B are required." >&2 && exit 1
[ "$MODEL_A_ID" = "$MODEL_B_ID" ] && echo "Model A and model B must be different." >&2 && exit 1

SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$CLUSTER_ID" '$1==cid')"
[ -z "$(trim "$SAMPLE_LINES")" ] && echo "No samples found for this cluster." >&2 && exit 1
SAMPLE_ROWS="$(offline_choose_multi "Select trajectories for delta-energy analysis:" "$SAMPLE_LINES")"
[ -z "$(trim "$SAMPLE_ROWS")" ] && echo "No samples selected." >&2 && exit 1

SAMPLE_IDS_ARR=()
FRAME_LIMIT_ARGS=()
while IFS= read -r row; do
  [ -z "$(trim "$row")" ] && continue
  sid="$(sample_id_from_row "$row" || true)"
  [ -z "$sid" ] && continue
  SAMPLE_IDS_ARR+=("$sid")
  n_frames="$(sample_frames_from_row "$row")"
  mode="$(prompt "Frames for ${sid} (${n_frames} available): all/random" "all")"
  mode="$(printf "%s" "$mode" | tr '[:upper:]' '[:lower:]')"
  if [ "$mode" = "random" ] || [ "$mode" = "r" ]; then
    limit="$(prompt "Random frames to use for ${sid}" "$n_frames")"
    if [[ "$limit" =~ ^[0-9]+$ ]] && [ "$limit" -gt 0 ] && [ "$limit" -lt "$n_frames" ]; then
      FRAME_LIMIT_ARGS+=(--frame-limit "${sid}:${limit}")
    fi
  fi
done <<< "$SAMPLE_ROWS"
[ "${#SAMPLE_IDS_ARR[@]}" -eq 0 ] && echo "No samples selected." >&2 && exit 1
SAMPLE_IDS="$(IFS=','; echo "${SAMPLE_IDS_ARR[*]}")"
echo "Selected ${#SAMPLE_IDS_ARR[@]} sample(s): $SAMPLE_IDS"

MD_LABEL_MODE="$(prompt "MD labels mode (assigned/halo)" "assigned")"
MD_LABEL_MODE="$(printf "%s" "$MD_LABEL_MODE" | tr '[:upper:]' '[:lower:]')"
[ "$MD_LABEL_MODE" != "halo" ] && MD_LABEL_MODE="assigned"
KEEP_INVALID="false"
if prompt_bool "Keep invalid frames? (y/N)" "N"; then KEEP_INVALID="true"; fi
ENERGY_BINS="$(prompt "Energy histogram bins" "80")"
SEED="$(prompt "Random seed" "0")"
WORKERS="$(prompt "Workers (0=auto)" "0")"
SHOW_PROGRESS="$(prompt "Show progress? (Y/n)" "Y")"
SHOW_PROGRESS="$(printf "%s" "$SHOW_PROGRESS" | tr '[:upper:]' '[:lower:]')"

CMD=(
  "$PYTHON_BIN" -m phase.scripts.potts_delta_energy
  --root "$OFFLINE_ROOT"
  --project-id "$OFFLINE_PROJECT_ID"
  --system-id "$OFFLINE_SYSTEM_ID"
  --cluster-id "$CLUSTER_ID"
  --model-a-id "$MODEL_A_ID"
  --model-b-id "$MODEL_B_ID"
  --sample-ids "$SAMPLE_IDS"
  --md-label-mode "$MD_LABEL_MODE"
  --energy-bins "$ENERGY_BINS"
  --seed "$SEED"
  --workers "$WORKERS"
)
CMD+=("${FRAME_LIMIT_ARGS[@]}")
[ "$KEEP_INVALID" = "true" ] && CMD+=(--keep-invalid)
[ "$SHOW_PROGRESS" != "n" ] && [ "$SHOW_PROGRESS" != "no" ] && CMD+=(--progress)

echo ""
printf 'Running: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
