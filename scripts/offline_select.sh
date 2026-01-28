#!/usr/bin/env bash
set -euo pipefail

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

trim() {
  printf "%s" "$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

offline_prompt_root() {
  local default_root="$1"
  OFFLINE_ROOT="$(prompt "Offline data root" "$default_root")"
  OFFLINE_ROOT="$(trim "$OFFLINE_ROOT")"
  if [ -z "$OFFLINE_ROOT" ]; then
    echo "Offline root is required." >&2
    return 1
  fi
  export PHASE_DATA_ROOT="$OFFLINE_ROOT"
}

_offline_list() {
  local cmd="$1"
  shift
  python -m phase.scripts.offline_browser --root "$OFFLINE_ROOT" "$cmd" "$@"
}

offline_choose_one() {
  local label="$1"
  local lines="$2"
  local -a entries
  readarray -t entries < <(printf "%s\n" "$lines" | awk 'NF')
  if [ "${#entries[@]}" -eq 0 ] || [ -z "${entries[0]}" ]; then
    echo ""; return 0
  fi
  echo "$label" >&2
  local i=1
  for line in "${entries[@]}"; do
    [ -z "$line" ] && continue
    local id="${line%%|*}"
    local rest="${line#*|}"
    local name="${rest%%|*}"
    echo "  [$i] $name ($id)" >&2
    i=$((i+1))
  done
  local choice
  if [ -t 0 ]; then
    read -r -p "Select number: " choice || true
  else
    read -r choice || true
  fi
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#entries[@]}" ]; then
    echo ""; return 0
  fi
  echo "${entries[$((choice-1))]}"
}

offline_choose_multi() {
  local label="$1"
  local lines="$2"
  local -a entries
  readarray -t entries < <(printf "%s\n" "$lines" | awk 'NF')
  if [ "${#entries[@]}" -eq 0 ] || [ -z "${entries[0]}" ]; then
    echo ""; return 0
  fi
  echo "$label" >&2
  local i=1
  for line in "${entries[@]}"; do
    [ -z "$line" ] && continue
    local id="${line%%|*}"
    local rest="${line#*|}"
    local name="${rest%%|*}"
    echo "  [$i] $name ($id)" >&2
    i=$((i+1))
  done
  local choice
  if [ -t 0 ]; then
    read -r -p "Select numbers (comma separated): " choice || true
  else
    read -r choice || true
  fi
  choice="$(trim "$choice")"
  if [ -z "$choice" ]; then
    echo ""; return 0
  fi
  local -a picks
  IFS=',' read -r -a picks <<< "$choice"
  local -a selected
  for pick in "${picks[@]}"; do
    pick="$(trim "$pick")"
    if [[ "$pick" =~ ^[0-9]+$ ]] && [ "$pick" -ge 1 ] && [ "$pick" -le "${#entries[@]}" ]; then
      selected+=("${entries[$((pick-1))]}")
    fi
  done
  printf "%s\n" "${selected[@]}"
}

offline_select_project() {
  local lines
  lines="$(_offline_list list-projects)"
  local selected
  selected="$(offline_choose_one "Available projects:" "$lines")"
  OFFLINE_PROJECT_ID="${selected%%|*}"
  OFFLINE_PROJECT_NAME=""
  if [ -n "$selected" ]; then
    OFFLINE_PROJECT_NAME="$(printf "%s" "$selected" | awk -F'|' '{print $2}')"
  fi
  if [ -z "$OFFLINE_PROJECT_ID" ]; then
    echo "No project selected." >&2
    return 1
  fi
  export OFFLINE_PROJECT_ID
  export OFFLINE_PROJECT_NAME
}

offline_select_system() {
  local lines
  lines="$(_offline_list list-systems --project-id "$OFFLINE_PROJECT_ID")"
  local selected
  selected="$(offline_choose_one "Available systems:" "$lines")"
  OFFLINE_SYSTEM_ID="${selected%%|*}"
  OFFLINE_SYSTEM_NAME=""
  if [ -n "$selected" ]; then
    OFFLINE_SYSTEM_NAME="$(printf "%s" "$selected" | awk -F'|' '{print $2}')"
  fi
  export OFFLINE_SYSTEM_ID
  export OFFLINE_SYSTEM_NAME
}

offline_select_cluster() {
  local lines
  lines="$(_offline_list list-clusters --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_one "Available clusters:" "$lines"
}

offline_select_model() {
  local lines
  lines="$(_offline_list list-models --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_one "Available Potts models:" "$lines"
}

offline_select_descriptors() {
  local lines
  lines="$(_offline_list list-descriptors --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_multi "Descriptor NPZ files:" "$lines"
}

offline_select_descriptor_one() {
  local lines
  lines="$(_offline_list list-descriptors --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_one "Descriptor NPZ file:" "$lines"
}

offline_select_pdbs() {
  local lines
  lines="$(_offline_list list-pdbs --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_multi "PDB files:" "$lines"
}

offline_select_states() {
  local lines
  lines="$(_offline_list list-states --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_multi "States:" "$lines"
}

offline_select_analysis_states() {
  local lines
  lines="$(_offline_list list-analysis-states --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_multi "States (macro + metastable):" "$lines"
}

offline_select_state_one() {
  local lines
  lines="$(_offline_list list-states --project-id "$OFFLINE_PROJECT_ID" --system-id "$OFFLINE_SYSTEM_ID")"
  offline_choose_one "State:" "$lines"
}
