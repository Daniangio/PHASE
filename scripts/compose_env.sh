#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/phase_defaults.sh"
ENV_FILE="${ROOT_DIR}/.env"

UID_VAL="$(id -u)"
GID_VAL="$(id -g)"
DEFAULTS_ROOT_VAL="$(phase_read_key "$(phase_defaults_file "$ROOT_DIR")" PHASE_DATA_ROOT)"
ENV_HOST_ROOT_VAL="$(phase_read_key "$ENV_FILE" PHASE_HOST_DATA_ROOT)"
ENV_DATA_ROOT_VAL="$(phase_read_key "$ENV_FILE" PHASE_DATA_ROOT)"
ENV_COMPOSE_PROJECT_NAME_VAL="$(phase_read_key "$ENV_FILE" COMPOSE_PROJECT_NAME)"
ENV_BACKEND_PORT_VAL="$(phase_read_key "$ENV_FILE" PHASE_BACKEND_PORT)"
ENV_FRONTEND_PORT_VAL="$(phase_read_key "$ENV_FILE" PHASE_FRONTEND_PORT)"
ENV_REDIS_PORT_VAL="$(phase_read_key "$ENV_FILE" PHASE_REDIS_PORT)"
USER_SLUG="$(id -un | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')"
if [ -z "$USER_SLUG" ]; then
  USER_SLUG="user"
fi
DEFAULT_COMPOSE_PROJECT_NAME="phase_${USER_SLUG}"
DATA_ROOT_VAL="${PHASE_HOST_DATA_ROOT:-${DEFAULTS_ROOT_VAL:-${PHASE_DATA_ROOT:-${ENV_HOST_ROOT_VAL:-${ENV_DATA_ROOT_VAL:-${ROOT_DIR}/data}}}}}"
COMPOSE_PROJECT_NAME_VAL="${COMPOSE_PROJECT_NAME:-${ENV_COMPOSE_PROJECT_NAME_VAL:-${DEFAULT_COMPOSE_PROJECT_NAME}}}"
BACKEND_PORT_VAL="${PHASE_BACKEND_PORT:-${ENV_BACKEND_PORT_VAL:-}}"
FRONTEND_PORT_VAL="${PHASE_FRONTEND_PORT:-${ENV_FRONTEND_PORT_VAL:-}}"
REDIS_PORT_VAL="${PHASE_REDIS_PORT:-${ENV_REDIS_PORT_VAL:-}}"
DOCKER_USER_VAL="${UID_VAL}:${GID_VAL}"

if [ -n "$DATA_ROOT_VAL" ]; then
  phase_ensure_writable_data_root "$DATA_ROOT_VAL"
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [ -f "$ENV_FILE" ]; then
  # Remove previous values (keep any other compose env vars intact).
  grep -v -E '^(COMPOSE_PROJECT_NAME|PHASE_UID|PHASE_GID|PHASE_DOCKER_USER|PHASE_HOST_DATA_ROOT|PHASE_DATA_ROOT|PHASE_BACKEND_PORT|PHASE_FRONTEND_PORT|PHASE_REDIS_PORT)=' "$ENV_FILE" > "$tmp" || true
else
  : > "$tmp"
fi

{
  echo "COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME_VAL}"
  echo "PHASE_UID=${UID_VAL}"
  echo "PHASE_GID=${GID_VAL}"
  echo "PHASE_DOCKER_USER=${DOCKER_USER_VAL}"
  if [ -n "$DATA_ROOT_VAL" ]; then
    echo "PHASE_HOST_DATA_ROOT=${DATA_ROOT_VAL}"
    echo "PHASE_DATA_ROOT=${DATA_ROOT_VAL}"
  fi
  if [ -n "$BACKEND_PORT_VAL" ]; then
    echo "PHASE_BACKEND_PORT=${BACKEND_PORT_VAL}"
  fi
  if [ -n "$FRONTEND_PORT_VAL" ]; then
    echo "PHASE_FRONTEND_PORT=${FRONTEND_PORT_VAL}"
  fi
  if [ -n "$REDIS_PORT_VAL" ]; then
    echo "PHASE_REDIS_PORT=${REDIS_PORT_VAL}"
  fi
} >> "$tmp"

mv "$tmp" "$ENV_FILE"
trap - EXIT

if [ -n "$DATA_ROOT_VAL" ]; then
  phase_write_key "$(phase_defaults_file "$ROOT_DIR")" PHASE_DATA_ROOT "$DATA_ROOT_VAL"
fi

echo "Wrote ${ENV_FILE}"
echo "  COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME_VAL}"
echo "  PHASE_UID=${UID_VAL}"
echo "  PHASE_GID=${GID_VAL}"
echo "  PHASE_DOCKER_USER=${DOCKER_USER_VAL}"
if [ -n "$DATA_ROOT_VAL" ]; then
  echo "  PHASE_DATA_ROOT=${DATA_ROOT_VAL}"
  echo "  PHASE_HOST_DATA_ROOT=${DATA_ROOT_VAL}"
else
  echo "  PHASE_DATA_ROOT not set; docker-compose will fall back to ./data"
fi
if [ -n "$FRONTEND_PORT_VAL" ]; then
  echo "  PHASE_FRONTEND_PORT=${FRONTEND_PORT_VAL}"
else
  echo "  PHASE_FRONTEND_PORT not set; docker-compose will use 18080"
fi
if [ -n "$BACKEND_PORT_VAL" ]; then
  echo "  PHASE_BACKEND_PORT=${BACKEND_PORT_VAL}"
else
  echo "  PHASE_BACKEND_PORT not set; docker-compose will use 18000"
fi
if [ -n "$REDIS_PORT_VAL" ]; then
  echo "  PHASE_REDIS_PORT=${REDIS_PORT_VAL}"
else
  echo "  PHASE_REDIS_PORT not set; docker-compose will use 16380"
fi
