#!/usr/bin/env bash
# Builds models/<name>'s ONNX from its pinned upstream: checkout, verified checkpoint, export, check.
set -euo pipefail

name="${1:?usage: build.sh <name> <out-dir>}"
out="${2:?usage: build.sh <name> <out-dir>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/antfrastructure.sh
source "${root}/scripts/lib/antfrastructure.sh"
antfrastructure_source linux/scripts/01-core/uv-image-overlay.sh
spec="${root}/models/${name}/model.json"
[ -f "${spec}" ] || { printf 'no %s\n' "${spec}" >&2; exit 2; }
# The image's venv holds the pinned CPython, torch, torchvision and the chain ONNX Runtime; elsewhere ORT would be PyPI's.
image_venv=/opt/venv
census=/opt/scripts/03-media/final/ort-venv-census.py
if [ -z "${ORT_CHAIN_WHEEL_DIR:-}" ] || [ ! -x "${image_venv}/bin/python" ] || [ ! -f "${census}" ]; then
  printf 'build.sh runs in the family Linux image (docs/models.md#building)\n' >&2
  exit 2
fi
work="${MODELANT_WORK:-${root}/.work}"
mkdir -p "${work}" "${out}"
# The export runs inside the upstream checkout, so a relative <out-dir> would land there.
out="$(cd "${out}" && pwd)"

field() { python3 -c "import json,sys; d=json.load(open('${spec}')); print(eval('d' + sys.argv[1]))" "$1"; }
repo="$(field "['upstream']['repository']")"
commit="$(field "['upstream']['commit']")"
config="$(field "['upstream']['config']")"
ckpt_file="$(field "['upstream']['checkpoint']['file']")"
ckpt_sha="$(field "['upstream']['checkpoint']['sha256']")"
drive_id="$(field "['upstream']['checkpoint']['google_drive_id']")"
file="$(field "['file']")"

venv="$(mktemp -d "${work}/venv.XXXXXXXXXX")"
trap 'uv_image_overlay_exit "$?" "${venv}"' EXIT
"${image_venv}/bin/python" -m venv --without-pip "${venv}"
# Image packages remain behind the fresh overlay in Python's import order.
"${image_venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' \
  > "$("${venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/modelant-image-venv.pth"
# Every ONNX Runtime the export can import must be a chain wheel from the image.
"${venv}/bin/python" -I "${census}" --check --store "${ORT_CHAIN_WHEEL_DIR}"
uv_image_overlay_verify_modules "${venv}/bin/python" "${image_venv}" torch torchvision onnxruntime numpy PIL yaml scipy
mkdir -p "${venv}/uv-overlay"
(cd "${root}" && uv_image_overlay_install "${image_venv}/bin/python" "${venv}/bin/python" "${venv}/uv-overlay")
uv_image_overlay_verify_modules "${venv}/bin/python" "${image_venv}" torch torchvision onnxruntime numpy PIL yaml scipy
"${venv}/bin/python" -I "${census}" --check --store "${ORT_CHAIN_WHEEL_DIR}"
# shellcheck disable=SC1091
. "${venv}/bin/activate"

src="${work}/upstream-${name}"
if [ ! -d "${src}/.git" ]; then
  git clone -q "${repo}" "${src}"
fi
git -C "${src}" fetch -q origin "${commit}" 2>/dev/null || true
git -C "${src}" checkout -q "${commit}"

ckpt="${work}/${ckpt_file}"
# The release's own mirror first, then the upstream's Google Drive.
if [ ! -s "${ckpt}" ] && [ -n "${MODELANT_CHECKPOINT_URL:-}" ]; then
  curl -fsSL -o "${ckpt}" "${MODELANT_CHECKPOINT_URL}" || rm -f "${ckpt}"
fi
[ -s "${ckpt}" ] || gdown -q "${drive_id}" -O "${ckpt}"
printf '%s  %s\n' "${ckpt_sha}" "${ckpt}" | sha256sum -c --quiet - || { printf 'checkpoint %s does not match its sha256\n' "${ckpt}" >&2; exit 1; }

(cd "${src}" && python3 "${root}/scripts/export_deim.py" --deim "${src}" --config "${config}" \
  --checkpoint "${ckpt}" --model-json "${spec}" --out "${out}/${file}")
cp "${ckpt}" "${out}/${ckpt_file}"
cp "${root}/models/${name}/LICENSE" "${out}/${name}.LICENSE"
cp "${root}/models/${name}/NOTICE" "${out}/${name}.NOTICE"
cp "${spec}" "${out}/${name}.model.json"
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "${root}" -cf - contracts \
  | gzip -n > "${out}/${name}.contract.tar.gz"
(cd "${out}" && sha256sum "${file}" "${name}.LICENSE" "${name}.NOTICE" "${name}.model.json" "${ckpt_file}" "${name}.contract.tar.gz" > SHA256SUMS && cat SHA256SUMS)
