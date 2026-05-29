#!/usr/bin/env bash
set -euo pipefail

# Repo-local defaults used by CLI helpers and Docker Compose.
# Docker Compose reads .env automatically; .phase_defaults keeps PHASE-specific
# defaults separate from other compose settings.

phase_defaults_file() {
  local root_dir="${1:?root dir required}"
  printf "%s/.phase_defaults" "$root_dir"
}

phase_compose_env_file() {
  local root_dir="${1:?root dir required}"
  printf "%s/.env" "$root_dir"
}

phase_read_key() {
  local file="$1"
  local key="$2"
  if [ ! -f "$file" ]; then
    return 0
  fi
  awk -v key="$key" '
    index($0, key "=") == 1 { value = substr($0, length(key) + 2) }
    END { if (value != "") print value }
  ' "$file"
}

phase_write_key() {
  local file="$1"
  local key="$2"
  local value="$3"
  local tmp
  tmp="$(mktemp)"
  if [ -f "$file" ]; then
    grep -v -E "^${key}=" "$file" > "$tmp" || true
  else
    : > "$tmp"
  fi
  printf "%s=%s\n" "$key" "$value" >> "$tmp"
  mv "$tmp" "$file"
}

phase_default_data_root() {
  local root_dir="${1:?root dir required}"
  local value="${PHASE_DATA_ROOT:-}"
  if [ -z "$value" ]; then
    value="$(phase_read_key "$(phase_defaults_file "$root_dir")" PHASE_DATA_ROOT)"
  fi
  if [ -z "$value" ]; then
    value="$(phase_read_key "$(phase_compose_env_file "$root_dir")" PHASE_HOST_DATA_ROOT)"
  fi
  if [ -z "$value" ]; then
    value="$(phase_read_key "$(phase_compose_env_file "$root_dir")" PHASE_DATA_ROOT)"
  fi
  if [ -z "$value" ]; then
    value="${root_dir}/data"
  fi
  printf "%s" "$value"
}

phase_ensure_writable_data_root() {
  local data_root="${1:?data root required}"
  mkdir -p "$data_root"
  local probe="${data_root}/.phase-write-test-$$"
  if ! ( : > "$probe" ) 2>/dev/null; then
    echo "PHASE data root is not writable by the current user: $data_root" >&2
    echo "Choose a directory owned by this user, fix permissions, or run:" >&2
    echo "  sudo chown -R $(id -u):$(id -g) '$data_root'" >&2
    return 1
  fi
  rm -f "$probe"
}

phase_persist_data_root() {
  local root_dir="${1:?root dir required}"
  local data_root="${2:?data root required}"
  data_root="$(printf "%s" "$data_root" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [ -z "$data_root" ]; then
    echo "Cannot persist an empty PHASE_DATA_ROOT." >&2
    return 1
  fi

  phase_ensure_writable_data_root "$data_root"

  local defaults_file env_file uid_val gid_val
  defaults_file="$(phase_defaults_file "$root_dir")"
  env_file="$(phase_compose_env_file "$root_dir")"
  uid_val="$(id -u)"
  gid_val="$(id -g)"

  phase_write_key "$defaults_file" PHASE_DATA_ROOT "$data_root"

  # Docker Compose reads this file automatically from the project directory.
  # Keep UID/GID here too so containers write files as the current host user.
  phase_write_key "$env_file" PHASE_HOST_DATA_ROOT "$data_root"
  phase_write_key "$env_file" PHASE_DATA_ROOT "$data_root"
  phase_write_key "$env_file" PHASE_UID "$uid_val"
  phase_write_key "$env_file" PHASE_GID "$gid_val"
  phase_write_key "$env_file" PHASE_DOCKER_USER "${uid_val}:${gid_val}"

  export PHASE_DATA_ROOT="$data_root"
}
