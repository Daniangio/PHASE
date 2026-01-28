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
USE_SLUG_IDS="false"

ensure_env() {
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    return 0
  fi
  echo "No active virtual environment detected."
  if [ -x "${DEFAULT_ENV}/bin/python" ]; then
    echo "Activate it with: source ${DEFAULT_ENV}/bin/activate"
  else
    echo "Create one first: scripts/potts_setup.sh"
  fi
  return 1
}

print_state_summary() {
  set +e
  local lines="$1"
  if [ -z "$lines" ]; then
    echo "  (no states)"
    set -e
    return
  fi
  while IFS='|' read -r col1 col2 col3 col4 col5; do
    [ -z "$col1" ] && continue
    echo "  - ${col2:-$col1} (${col1})"
    [ -n "$col3" ] && echo "      pdb: $col3"
    [ -n "$col4" ] && echo "      traj: $col4"
    [ -n "$col5" ] && echo "      descriptors: $col5"
  done <<< "$lines"
  set -e
}

print_simple_list() {
  set +e
  local lines="$1"
  if [ -z "$lines" ]; then
    echo "  (none)"
    set -e
    return
  fi
  while IFS='|' read -r col1 col2 col3 col4; do
    [ -z "$col1" ] && continue
    echo "  - ${col2:-$col1} (${col1})"
    [ -n "$col3" ] && echo "      path: $col3"
    [ -n "$col4" ] && echo "      data: $col4"
  done <<< "$lines"
  set -e
}

pause() {
  read -r -p "Press Enter to continue..." _ || true
}

project_menu() {
  while true; do
    echo ""
    echo "Projects:"
    python -m phase.scripts.offline_browser --root "$OFFLINE_ROOT" list-projects || true
    echo ""
    ACTION_LINES=$'open|Open project\nnew|Create project\nmigrate|Migrate IDs to name slugs\nrefresh|Refresh list\nquit|Quit'
    ACTION_ROW="$(offline_choose_one "Project actions:" "$ACTION_LINES")"
    ACTION="$(printf "%s" "$ACTION_ROW" | awk -F'|' '{print $1}')"
    case "$ACTION" in
      open)
        if offline_select_project; then
          system_menu
        fi
        ;;
      new)
        NAME="$(prompt "Project name" "Project")"
        DESC="$(prompt "Project description" "")"
        EXTRA_ARGS=()
        if [ "$USE_SLUG_IDS" = "true" ]; then
          EXTRA_ARGS+=(--use-slug-ids)
        fi
        python -m phase.scripts.offline_system --root "$OFFLINE_ROOT" init-project --name "$NAME" ${DESC:+--description "$DESC"} "${EXTRA_ARGS[@]}"
        ;;
      migrate)
        echo ""
        echo "Planned migrations:"
        python -m phase.scripts.offline_migrate --root "$OFFLINE_ROOT" --dry-run || true
        if prompt_bool "Apply these changes? (y/N)" "N"; then
          python -m phase.scripts.offline_migrate --root "$OFFLINE_ROOT"
          echo "Migration complete."
        else
          echo "No changes made."
        fi
        ;;
      refresh) ;;
      quit|"")
        return 0
        ;;
    esac
  done
}

system_menu() {
  while true; do
    echo ""
    echo "Project: ${OFFLINE_PROJECT_NAME:-$OFFLINE_PROJECT_ID} (${OFFLINE_PROJECT_ID})"
    echo "Systems:"
    python -m phase.scripts.offline_browser --root "$OFFLINE_ROOT" list-systems --project-id "$OFFLINE_PROJECT_ID" || true
    echo ""
    ACTION_LINES=$'open|Open system\nnew|Create system\nback|Back to projects'
    ACTION_ROW="$(offline_choose_one "System actions:" "$ACTION_LINES")"
    ACTION="$(printf "%s" "$ACTION_ROW" | awk -F'|' '{print $1}')"
    case "$ACTION" in
      open)
        if offline_select_system; then
          state_menu
        fi
        ;;
      new)
        NAME="$(prompt "System name" "System")"
        DESC="$(prompt "System description" "")"
        EXTRA_ARGS=()
        if [ "$USE_SLUG_IDS" = "true" ]; then
          EXTRA_ARGS+=(--use-slug-ids)
        fi
        python -m phase.scripts.offline_system --root "$OFFLINE_ROOT" create-system --project-id "$OFFLINE_PROJECT_ID" --name "$NAME" ${DESC:+--description "$DESC"} "${EXTRA_ARGS[@]}"
        ;;
      back|"")
        return 0
        ;;
    esac
  done
}

state_menu() {
  while true; do
    echo ""
    echo "Project: ${OFFLINE_PROJECT_NAME:-$OFFLINE_PROJECT_ID} (${OFFLINE_PROJECT_ID})"
    echo "System: ${OFFLINE_SYSTEM_NAME:-$OFFLINE_SYSTEM_ID} (${OFFLINE_SYSTEM_ID})"
    echo ""
    echo "States:"
    STATE_LINES="$(_offline_list list-states --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
    print_state_summary "$STATE_LINES"
    echo ""
    ACTION_LINES=$'add-state|Add state (PDB+trajectory)\nlist-clusters|List clusters\nlist-models|List Potts models\nlist-samples|List sampling runs\nclean-models|Clean stale Potts model entries\ncluster|Run clustering\nfit|Fit Potts model\nsample|Run sampling\nback|Back to systems'
    ACTION_ROW="$(offline_choose_one "System actions:" "$ACTION_LINES")"
    ACTION="$(printf "%s" "$ACTION_ROW" | awk -F'|' '{print $1}')"
    case "$ACTION" in
      add-state)
        STATE_ID="$(prompt "State ID" "state_1")"
        STATE_NAME="$(prompt "State name" "$STATE_ID")"
        PDB_PATH="$(prompt "PDB path" "")"
        TRAJ_PATH="$(prompt "Trajectory path" "")"
        SLICE_SPEC="$(prompt "Frame slice start:stop:step (blank = full; number = step)" "")"
        RES_SEL="$(prompt "Residue selection (optional)" "")"
        COPY_TRAJ="false"
        if prompt_bool "Copy trajectory into system folder? (y/N)" "N"; then
          COPY_TRAJ="true"
        fi
        ensure_env || return 0
        CMD=(python -m phase.scripts.offline_system --root "$OFFLINE_ROOT" add-state \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID" \
          --state-id "$STATE_ID" \
          --name "$STATE_NAME" \
          --pdb "$PDB_PATH" \
          --traj "$TRAJ_PATH")
        if [ "$COPY_TRAJ" = "true" ]; then
          CMD+=(--copy-traj)
        fi
        if [ -n "$SLICE_SPEC" ]; then
          CMD+=(--slice-spec "$SLICE_SPEC")
        fi
        if [ -n "$RES_SEL" ]; then
          CMD+=(--residue-selection "$RES_SEL")
        fi
        if ! "${CMD[@]}"; then
          echo "Add state failed. Check paths and try again."
        fi
        pause
        ;;
      list-clusters)
        echo ""
        echo "Clusters:"
        CLUSTER_LINES="$(_offline_list list-clusters --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
        print_simple_list "$CLUSTER_LINES"
        pause
        ;;
      list-models)
        echo ""
        echo "Potts models:"
        MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
        print_simple_list "$MODEL_LINES"
        pause
        ;;
      clean-models)
        echo ""
        echo "Stale model entries (missing files):"
        STALE_LINES="$(_offline_list prune-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" --dry-run || true)"
        print_simple_list "$STALE_LINES"
        if prompt_bool "Remove these entries from metadata? (y/N)" "N"; then
          _offline_list prune-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" >/dev/null
          echo "Cleaned."
        else
          echo "No changes made."
        fi
        pause
        ;;
      list-samples)
        echo ""
        echo "Sampling runs:"
        SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
        print_simple_list "$SAMPLE_LINES"
        pause
        ;;
      cluster)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/cluster_npz.sh" \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID"
        ;;
      fit)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/potts_fit.sh" \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID"
        ;;
      sample)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/potts_sampling.sh"
        ;;
      back|"")
        return 0
        ;;
    esac
  done
}

offline_prompt_root "${ROOT_DIR}/data"
USE_SLUG_IDS="true"
project_menu
