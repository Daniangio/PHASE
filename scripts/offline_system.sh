#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/offline_select.sh"

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

offline_prompt_root "${ROOT_DIR}/data"

ACTION_LINES=$'init-project|Initialize a new project\ncreate-system|Create a new system\nadd-state|Add a state (PDB+trajectory)\nlist|List projects/systems'
ACTION_ROW="$(offline_choose_one "Select action:" "$ACTION_LINES")"
MODE="$(printf "%s" "$ACTION_ROW" | awk -F'|' '{print $1}')"
MODE="$(printf "%s" "$MODE" | tr '[:upper:]' '[:lower:]')"

if [ "$MODE" = "init-project" ]; then
  NAME="$(prompt "Project name" "Project")"
  DESC="$(prompt "Project description" "")"
  EXTRA_ARGS=()
  EXTRA_ARGS+=(--use-slug-ids)
  python -m phase.scripts.offline_system --root "$OFFLINE_ROOT" init-project --name "$NAME" ${DESC:+--description "$DESC"} "${EXTRA_ARGS[@]}"
  exit 0
fi

if [ "$MODE" = "create-system" ]; then
  if ! offline_select_project; then
    echo "No projects found. Run init-project first." >&2
    exit 1
  fi
  NAME="$(prompt "System name" "System")"
  DESC="$(prompt "System description" "")"
  EXTRA_ARGS=()
  EXTRA_ARGS+=(--use-slug-ids)
  python -m phase.scripts.offline_system --root "$OFFLINE_ROOT" create-system --project-id "$OFFLINE_PROJECT_ID" --name "$NAME" ${DESC:+--description "$DESC"} "${EXTRA_ARGS[@]}"
  exit 0
fi

if [ "$MODE" = "add-state" ]; then
  if ! offline_select_project; then
    echo "No projects found. Run init-project first." >&2
    exit 1
  fi
  offline_select_system
  STATE_ID="$(prompt "State ID" "state_1")"
  STATE_NAME="$(prompt "State name" "$STATE_ID")"
  PDB_PATH="$(prompt "PDB path" "")"
  TRAJ_PATH="$(prompt "Trajectory path" "")"
  SLICE_SPEC="$(prompt "Frame slice start:stop:step (blank = full; number = step)" "")"
  RES_SEL="$(prompt "Residue selection (optional)" "")"
  COPY_TRAJ_ARGS=()
  if prompt_bool "Copy trajectory into system folder? (y/N)" "N"; then
    COPY_TRAJ_ARGS+=(--copy-traj)
  fi
  python -m phase.scripts.offline_system --root "$OFFLINE_ROOT" add-state \
  --project-id "$OFFLINE_PROJECT_ID" \
  --system-id "$OFFLINE_SYSTEM_ID" \
  --state-id "$STATE_ID" \
  --name "$STATE_NAME" \
  --pdb "$PDB_PATH" \
  --traj "$TRAJ_PATH" \
  ${RES_SEL:+--residue-selection "$RES_SEL"} \
  ${SLICE_SPEC:+--slice-spec "$SLICE_SPEC"} \
  "${COPY_TRAJ_ARGS[@]}"
  exit 0
fi

if [ "$MODE" = "list" ]; then
  echo "Projects:"
  python -m phase.scripts.offline_browser --root "$OFFLINE_ROOT" list-projects
  if prompt_bool "Show details for a project/system? (y/N)" "N"; then
    offline_select_project
    offline_select_system

    echo ""
    echo "Project: ${OFFLINE_PROJECT_NAME:-$OFFLINE_PROJECT_ID} (${OFFLINE_PROJECT_ID})"
    echo "System: ${OFFLINE_SYSTEM_NAME:-$OFFLINE_SYSTEM_ID} (${OFFLINE_SYSTEM_ID})"

    print_section() {
      local title="$1"
      local lines="$2"
      local mode="$3"
      echo ""
      echo "$title"
      if [ -z "$lines" ]; then
        echo "  (none)"
        return
      fi
      while IFS='|' read -r col1 col2 col3 col4 col5; do
        [ -z "$col1" ] && continue
        case "$mode" in
          states)
            echo "  - ${col2:-$col1} (${col1})"
            [ -n "$col3" ] && echo "      pdb: $col3"
            [ -n "$col4" ] && echo "      traj: $col4"
            [ -n "$col5" ] && echo "      descriptors: $col5"
            ;;
          clusters)
            echo "  - ${col2:-$col1} (${col1})"
            [ -n "$col3" ] && echo "      npz: $col3"
            ;;
          models)
            echo "  - ${col2:-$col1} (cluster ${col1})"
            [ -n "$col3" ] && echo "      model: $col3"
            ;;
          descriptors)
            echo "  - ${col2:-$col1} (${col1})"
            [ -n "$col3" ] && echo "      npz: $col3"
            ;;
          sampling)
            echo "  - ${col2:-$col1} (${col1})"
            [ -n "$col3" ] && echo "      summary: $col3"
            [ -n "$col4" ] && echo "      data: $col4"
            ;;
          *)
            echo "  - ${col2:-$col1} (${col1})"
            ;;
        esac
      done <<< "$lines"
    }

    STATE_LINES="$(_offline_list list-states --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
    DESC_LINES="$(_offline_list list-descriptors --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
    CLUSTER_LINES="$(_offline_list list-clusters --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
    MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
    SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"

    print_section "States" "$STATE_LINES" "states"
    print_section "Descriptors" "$DESC_LINES" "descriptors"
    print_section "Clusters" "$CLUSTER_LINES" "clusters"
    print_section "Potts models" "$MODEL_LINES" "models"
    print_section "Sampling runs" "$SAMPLE_LINES" "sampling"
  fi
  exit 0
fi

echo "Unknown mode: $MODE"
exit 1
