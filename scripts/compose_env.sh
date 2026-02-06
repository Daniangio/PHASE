#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

UID_VAL="$(id -u)"
GID_VAL="$(id -g)"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

if [ -f "$ENV_FILE" ]; then
  # Remove previous values (keep any other compose env vars intact).
  grep -v -E '^(PHASE_UID|PHASE_GID)=' "$ENV_FILE" > "$tmp" || true
else
  : > "$tmp"
fi

{
  echo "PHASE_UID=${UID_VAL}"
  echo "PHASE_GID=${GID_VAL}"
} >> "$tmp"

mv "$tmp" "$ENV_FILE"
trap - EXIT

echo "Wrote ${ENV_FILE}"
echo "  PHASE_UID=${UID_VAL}"
echo "  PHASE_GID=${GID_VAL}"

