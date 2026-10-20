#!/usr/bin/env bash
# Builds models/<name>'s ONNX from its pinned upstream: checkout, verified checkpoint, export, check.
# Usage: scripts/build.sh <name> <out-dir>   (in the family's Linux image, needs network; writes <out-dir>/<file>)
set -euo pipefail

name="${1:?usage: build.sh <name> <out-dir>}"
out="${2:?usage: build.sh <name> <out-dir>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spec="${root}/models/${name}/model.json"
[ -f "${spec}" ] || { printf 'no %s\n' "${spec}" >&2; exit 2; }
# The image's venv holds the pinned CPython, torch, torchvision and the chain ONNX Runtime; elsewhere ORT would be PyPI's.
image_venv=/opt/venv
census=/opt/scripts/03-media/final/ort-venv-census.py
if [ -z "${ORT_CHAIN_WHEEL_DIR:-}" ] || [ ! -x "${image_venv}/bin/python" ] || [ ! -f "${census}" ]; then
  printf 'build.sh runs in ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest (docs/models.md#building)\n' >&2
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

venv="${work}/venv"
if [ ! -x "${venv}/bin/python" ]; then
  "${image_venv}/bin/python" -m venv --without-pip "${venv}"
  # The image's packages sit behind the venv's own, so pip adds only what the export lacks.
  "${image_venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' \
    > "$("${venv}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/modelant-image-venv.pth"
  "${image_venv}/bin/python" -m pip list --format=freeze > "${work}/image-constraints.txt"
  # Held at the image's versions, so no requirement replaces its torch, numpy or ORT.
  "${venv}/bin/python" -m pip install -q -c "${work}/image-constraints.txt" -r "${root}/scripts/requirements.txt"
fi
# Every ONNX Runtime the export can import must be a chain wheel from the image.
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
(cd "${out}" && sha256sum "${file}" "${name}.LICENSE" "${name}.NOTICE" "${name}.model.json" "${ckpt_file}" > SHA256SUMS && cat SHA256SUMS)
