#!/usr/bin/env bash
# Builds models/<name>'s ONNX from its pinned upstream: checkout, verified checkpoint, export, check.
# Usage: scripts/build.sh <name> <out-dir>   (needs python3, git and network; writes <out-dir>/<file>)
set -euo pipefail

name="${1:?usage: build.sh <name> <out-dir>}"
out="${2:?usage: build.sh <name> <out-dir>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
spec="${root}/models/${name}/model.json"
[ -f "${spec}" ] || { printf 'no %s\n' "${spec}" >&2; exit 2; }
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

if [ ! -x "${work}/venv/bin/python" ]; then
  python3 -m venv "${work}/venv"
  # torch from the CPU index alone: PyPI's x86_64 wheel pulls in CUDA.
  "${work}/venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
  "${work}/venv/bin/pip" install -q -r "${root}/scripts/requirements.txt"
fi
# shellcheck disable=SC1091
. "${work}/venv/bin/activate"

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
