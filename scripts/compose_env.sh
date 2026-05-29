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
DATA_ROOT_VAL="${PHASE_HOST_DATA_ROOT:-${DEFAULTS_ROOT_VAL:-${PHASE_DATA_ROOT:-${ENV_HOST_ROOT_VAL:-${ENV_DATA_ROOT_VAL:-${ROOT_DIR}/data}}}}}"
BACKEND_PORT_VAL="${PHASE_BACKEND_PORT:-}"
FRONTEND_PORT_VAL="${PHASE_FRONTEND_PORT:-}"
REDIS_PORT_VAL="${PHASE_REDIS_PORT:-}"
DOCKER_USER_VAL="${UID_VAL}:${GID_VAL}"

if [ -n "$DATA_ROOT_VAL" ]; then
  phase_ensure_writable_data_root "$DATA_ROOT_VAL"
fi

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [ -f "$ENV_FILE" ]; then
  # Remove previous values (keep any other compose env vars intact).
  grep -v -E '^(PHASE_UID|PHASE_GID|PHASE_DOCKER_USER|PHASE_HOST_DATA_ROOT|PHASE_DATA_ROOT|PHASE_BACKEND_PORT|PHASE_FRONTEND_PORT|PHASE_REDIS_PORT)=' "$ENV_FILE" > "$tmp" || true
else
  : > "$tmp"
fi

{
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
  echo "  PHASE_FRONTEND_PORT not exported; docker-compose will use 18080"
fi
if [ -n "$BACKEND_PORT_VAL" ]; then
  echo "  PHASE_BACKEND_PORT=${BACKEND_PORT_VAL}"
else
  echo "  PHASE_BACKEND_PORT not exported; docker-compose will use 18000"
fi
if [ -n "$REDIS_PORT_VAL" ]; then
  echo "  PHASE_REDIS_PORT=${REDIS_PORT_VAL}"
else
  echo "  PHASE_REDIS_PORT not exported; docker-compose will use 16380"
fi
