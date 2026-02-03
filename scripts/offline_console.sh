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
DEFAULT_ROOT="${PHASE_DATA_ROOT:-${ROOT_DIR}/data}"

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

print_model_list() {
  set +e
  local lines="$1"
  if [ -z "$lines" ]; then
    echo "  (none)"
    set -e
    return
  fi
  while IFS='|' read -r model_id name path cluster_id; do
    [ -z "$model_id" ] && continue
    echo "  - ${name:-$model_id} (${model_id})"
    [ -n "$cluster_id" ] && echo "      cluster: ${cluster_id}"
    [ -n "$path" ] && echo "      path: $path"
  done <<< "$lines"
  set -e
}

print_sample_list() {
  set +e
  local lines="$1"
  if [ -z "$lines" ]; then
    echo "  (none)"
    set -e
    return
  fi
  while IFS='|' read -r cluster_id name sample_type path; do
    [ -z "$cluster_id" ] && continue
    echo "  - ${name} (${sample_type})"
    echo "      cluster: ${cluster_id}"
    [ -n "$path" ] && echo "      path: $path"
  done <<< "$lines"
  set -e
}

export_projects_zip() {
  if [ -z "${OFFLINE_ROOT:-}" ]; then
    echo "Offline root not set."
    return 1
  fi
  local projects_dir="${OFFLINE_ROOT}/projects"
  if [ ! -d "$projects_dir" ]; then
    echo "Projects directory not found at: ${projects_dir}"
    return 1
  fi
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local default_path="${OFFLINE_ROOT}/projects_dump_${ts}.zip"
  local out_path
  out_path="$(prompt "Export zip path" "$default_path")"
  out_path="$(trim "$out_path")"
  if [ -z "$out_path" ]; then
    echo "Export canceled."
    return 0
  fi
  python - <<PY
import pathlib
import shutil

root = pathlib.Path(r"$OFFLINE_ROOT")
out_path = pathlib.Path(r"$out_path")
base_name = out_path
if out_path.suffix.lower() == ".zip":
    base_name = out_path.with_suffix("")
base_name.parent.mkdir(parents=True, exist_ok=True)
archive = shutil.make_archive(str(base_name), "zip", root_dir=str(root), base_dir="projects")
print(f"Projects archive saved to: {archive}")
PY
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
    ACTION_LINES=$'open|Open project\nnew|Create project\nmigrate|Migrate IDs to name slugs\nexport|Export projects zip\nrefresh|Refresh list\nquit|Quit'
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
      export)
        export_projects_zip
        pause
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

cluster_menu() {
  while true; do
    echo ""
    echo "Project: ${OFFLINE_PROJECT_NAME:-$OFFLINE_PROJECT_ID} (${OFFLINE_PROJECT_ID})"
    echo "System: ${OFFLINE_SYSTEM_NAME:-$OFFLINE_SYSTEM_ID} (${OFFLINE_SYSTEM_ID})"
    echo "Cluster: ${OFFLINE_CLUSTER_NAME:-$OFFLINE_CLUSTER_ID} (${OFFLINE_CLUSTER_ID})"
    echo ""
    ACTION_LINES=$'list-models|List Potts models\nlist-samples|List sampling runs\nfit|Fit Potts model\nfit-delta|Fit delta Potts model\nsample|Run sampling\nevaluate|Evaluate state against cluster\nback|Back to systems'
    ACTION_ROW="$(offline_choose_one "Cluster actions:" "$ACTION_LINES")"
    ACTION="$(printf "%s" "$ACTION_ROW" | awk -F'|' '{print $1}')"
    case "$ACTION" in
      list-models)
        echo ""
        echo "Potts models:"
        MODEL_LINES="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
        MODEL_LINES="$(printf "%s\n" "$MODEL_LINES" | awk -F'|' -v cid="$OFFLINE_CLUSTER_ID" '$4==cid')"
        print_model_list "$MODEL_LINES"
        pause
        ;;
      list-samples)
        echo ""
        echo "Samples:"
        SAMPLE_LINES="$(_offline_list list-sampling --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID" || true)"
        SAMPLE_LINES="$(printf "%s\n" "$SAMPLE_LINES" | awk -F'|' -v cid="$OFFLINE_CLUSTER_ID" '$1==cid')"
        print_sample_list "$SAMPLE_LINES"
        pause
        ;;
      fit)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/potts_fit.sh" \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID" \
          --cluster-id "$OFFLINE_CLUSTER_ID" \
          --npz "$OFFLINE_CLUSTER_PATH"
        ;;
      fit-delta)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/potts_delta_fit.sh" \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID" \
          --cluster-id "$OFFLINE_CLUSTER_ID"
        ;;
      sample)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/potts_sampling.sh" \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID" \
          --cluster-id "$OFFLINE_CLUSTER_ID" \
          --npz "$OFFLINE_CLUSTER_PATH"
        ;;
      evaluate)
        ensure_env || return 0
        STATE_ROW="$(offline_select_state_one)"
        STATE_ID="$(printf "%s" "$STATE_ROW" | awk -F'|' '{print $1}')"
        if [ -z "$STATE_ID" ]; then
          echo "No state selected."
        fi
        python -m phase.scripts.evaluate_state \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID" \
          --cluster-id "$OFFLINE_CLUSTER_ID" \
          --state-id "$STATE_ID"
        pause
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
    ACTION_LINES=$'open-cluster|Open cluster\nadd-state|Add state (PDB+trajectory)\nlist-clusters|List clusters\ncluster|Run clustering\nback|Back to systems'
    ACTION_ROW="$(offline_choose_one "System actions:" "$ACTION_LINES")"
    ACTION="$(printf "%s" "$ACTION_ROW" | awk -F'|' '{print $1}')"
    case "$ACTION" in
      open-cluster)
        CLUSTER_ROW="$(offline_select_cluster)"
        OFFLINE_CLUSTER_ID="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $1}')"
        OFFLINE_CLUSTER_NAME="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $2}')"
        OFFLINE_CLUSTER_PATH="$(printf "%s" "$CLUSTER_ROW" | awk -F'|' '{print $3}')"
        if [ -z "$OFFLINE_CLUSTER_ID" ]; then
          echo "No cluster selected."
        else
          export OFFLINE_CLUSTER_ID OFFLINE_CLUSTER_NAME OFFLINE_CLUSTER_PATH
          cluster_menu
        fi
        ;;
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
      cluster)
        ensure_env || return 0
        "${ROOT_DIR}/scripts/cluster_npz.sh" \
          --root "$OFFLINE_ROOT" \
          --project-id "$OFFLINE_PROJECT_ID" \
          --system-id "$OFFLINE_SYSTEM_ID"
        ;;
      back|"")
        return 0
        ;;
    esac
  done
}

  offline_prompt_root "${DEFAULT_ROOT}"
USE_SLUG_IDS="true"
project_menu
