#!/usr/bin/env bash
# Exercise the real overlay against the family image without writing its environment.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/antfrastructure.sh
source "${root}/scripts/lib/antfrastructure.sh"
antfrastructure_source linux/scripts/01-core/uv-image-overlay.sh
work="$(mktemp -d)"
trap 'uv_image_overlay_exit "$?" "${work}"' EXIT
/opt/venv/bin/python -m venv --without-pip "${work}/venv"
/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))' > "$("${work}/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/modelant-image-venv.pth"
uv_image_overlay_verify_modules "${work}/venv/bin/python" /opt/venv torch torchvision onnxruntime numpy PIL yaml scipy
"${work}/venv/bin/python" -I /opt/scripts/03-media/final/ort-venv-census.py --check --store "${ORT_CHAIN_WHEEL_DIR:?}"
mkdir -p "${work}/overlay"
(cd "${root}" && uv_image_overlay_install /opt/venv/bin/python "${work}/venv/bin/python" "${work}/overlay")
uv_image_overlay_verify_modules "${work}/venv/bin/python" /opt/venv torch torchvision onnxruntime numpy PIL yaml scipy
"${work}/venv/bin/python" -I /opt/scripts/03-media/final/ort-venv-census.py --check --store "${ORT_CHAIN_WHEEL_DIR:?}"
"${work}/venv/bin/python" -c 'import accelerate, calflops, onnx, tensorboard, transformers'
