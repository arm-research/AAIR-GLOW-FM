#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  printf '[setup] %s\n' "$*"
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

create_venv() {
  if [[ -d "${VENV_DIR}" ]]; then
    log "Refreshing virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"
  else
    log "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi

  "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel
  install_cuda_torch_bootstrap
  install_build_bootstrap
  install_requirements
  verify_runtime
}

install_cuda_torch_bootstrap() {
  log "Bootstrapping pinned Torch packages required by CUDA extension builds"
  "${VENV_DIR}/bin/pip" install \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    torchaudio==2.7.1+cu126
}

install_build_bootstrap() {
  # flash-attn imports psutil while generating its package metadata. Installing
  # the pinned build-time dependency first is required even though it is also
  # declared in requirements.txt; pip resolves that file before installing it.
  log "Bootstrapping pinned requirements needed by extension metadata"
  "${VENV_DIR}/bin/pip" install psutil==7.2.2
}

install_requirements() {
  log "Installing pinned requirements"
  "${VENV_DIR}/bin/pip" install \
    --no-build-isolation \
    -r "${ROOT_DIR}/requirements.txt"
}

verify_runtime() {
  log "Verifying Torch, CUDA, flash_attn, and NATTEN runtime bindings"
  "${VENV_DIR}/bin/python" - <<'PY'
import flash_attn
import natten
import torch
import torchaudio
import torchvision

expected_versions = {
    "torch": (torch.__version__, "2.7.1+cu126"),
    "torchvision": (torchvision.__version__, "0.22.1+cu126"),
    "torchaudio": (torchaudio.__version__, "2.7.1+cu126"),
}
for package, (actual, expected) in expected_versions.items():
    if actual != expected:
        raise SystemExit(f"Unexpected {package} version: {actual} (expected {expected})")

if torch.version.cuda != "12.6":
    raise SystemExit(f"Unexpected Torch CUDA version: {torch.version.cuda} (expected 12.6)")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to Torch")

if flash_attn.__version__ != "2.8.3":
    raise SystemExit(f"Unexpected flash_attn version: {flash_attn.__version__}")

if natten.__version__ != "0.21.5.post1":
    raise SystemExit(f"Unexpected NATTEN version: {natten.__version__}")

if not natten.HAS_LIBNATTEN:
    raise SystemExit(
        "NATTEN was installed without libnatten support. Re-run setup after fixing the build environment."
    )

print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"torchvision={torchvision.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"natten={natten.__version__} HAS_LIBNATTEN={natten.HAS_LIBNATTEN}")
PY
}

main() {
  need_cmd "${PYTHON_BIN}"
  need_cmd git
  need_cmd nvcc

  create_venv

  log "Setup complete. Arrange DLL manually, then activate with: source ${VENV_DIR}/bin/activate"
}

main "$@"
