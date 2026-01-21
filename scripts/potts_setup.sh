#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PY="3.11"
if [ -d "${ROOT_DIR}/.venv-potts-fit" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
elif [ -d "${ROOT_DIR}/.venv" ]; then
  DEFAULT_ENV="${ROOT_DIR}/.venv"
else
  DEFAULT_ENV="${ROOT_DIR}/.venv-potts-fit"
fi

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

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

PY_VER="$(prompt "Python version" "${DEFAULT_PY}")"
VENV_DIR="$(prompt "Virtual env directory" "${DEFAULT_ENV}")"

if [ -x "${VENV_DIR}/bin/python" ]; then
  echo "Using existing virtual environment at: ${VENV_DIR}"
else
  if ! uv venv "${VENV_DIR}" --python "${PY_VER}"; then
    echo "Failed to create venv. Trying to install Python ${PY_VER} via uv..."
    uv python install "${PY_VER}"
    uv venv "${VENV_DIR}" --python "${PY_VER}"
  fi
fi

source "${VENV_DIR}/bin/activate"

REQ_TMP="$(mktemp)"
CONSTRAINTS_TMP="$(mktemp)"
trap 'rm -f "$REQ_TMP" "$CONSTRAINTS_TMP"' EXIT

grep -v -E '^torch($|[<>=])' "${ROOT_DIR}/requirements.txt" > "$REQ_TMP"
echo "numpy<2" > "$CONSTRAINTS_TMP"

echo "Installing base dependencies (numpy)..."
uv pip install -r "$CONSTRAINTS_TMP"

INSTALL_FULL="$(prompt "Install full PHASE deps (y/N)" "N")"
if [[ "$INSTALL_FULL" =~ ^[Yy]$ ]]; then
  echo "Installing full dependencies (excluding torch)..."
  uv pip install -r "$REQ_TMP" --constraints "$CONSTRAINTS_TMP"
fi

INSTALL_TORCH="$(prompt "Install torch now (y/N)" "Y")"
if [[ "$INSTALL_TORCH" =~ ^[Yy]$ ]]; then
  CUDA_DEFAULT="cpu"
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_DEFAULT="cu128"
  fi
  CUDA_FLAVOR="$(prompt "Torch CUDA flavor (cpu/cu118/cu121/cu124/cu128)" "${CUDA_DEFAULT}")"
  TORCH_VERSION="$(prompt "Torch version (blank = latest)" "")"
  TORCH_SPEC="torch"
  if [ -n "$TORCH_VERSION" ]; then
    TORCH_SPEC="torch==${TORCH_VERSION}"
  fi
  echo "Installing torch (${CUDA_FLAVOR})..."
  uv pip install "$TORCH_SPEC" --torch-backend "${CUDA_FLAVOR}"
fi

echo "Installing phase package (no deps)..."
uv pip install -e "${ROOT_DIR}" --no-deps

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
