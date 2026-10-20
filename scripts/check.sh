#!/usr/bin/env bash
# Every model is complete and every script parses, on the image's pinned CPython (check.yml runs it).
# Usage: scripts/check.sh   (in the family's Linux image)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
# The pinned CPython lives in the image's venv; the runner's own python3 is another version.
if [ ! -x /opt/venv/bin/python3 ] || [ -z "${PYTHON_MAJOR_MINOR:-}" ]; then
  printf 'check.sh runs in the family Linux image (docs/models.md#building)\n' >&2
  exit 2
fi
python=/opt/venv/bin/python3
printf 'python: %s\n' "$("${python}" --version)"

"${python}" -I - <<'PY'
import json, pathlib, re, sys
bad = []
for d in sorted(pathlib.Path("models").iterdir()):
    for f in ("model.json", "LICENSE", "NOTICE"):
        if not (d / f).is_file():
            bad.append(f"{d}: no {f}")
    spec = json.loads((d / "model.json").read_text())
    if spec["name"] != d.name:
        bad.append(f"{d}: name is {spec['name']}")
    if not re.fullmatch(r"[0-9a-f]{40}", spec["upstream"]["commit"]):
        bad.append(f"{d}: upstream commit is not a full sha")
    if not re.fullmatch(r"[0-9a-f]{64}", spec["upstream"]["checkpoint"]["sha256"]):
        bad.append(f"{d}: checkpoint sha256 is not 64 hex digits")
    print(f"{d.name}: {spec['license']}, {spec['upstream']['repository']}")
sys.exit("\n".join(bad) if bad else 0)
PY

bash -n scripts/build.sh scripts/check.sh
shellcheck -S warning scripts/build.sh scripts/check.sh
# compile(), not py_compile: nothing is written into the checkout.
"${python}" -I -c 'import sys; [compile(open(f, encoding="utf-8").read(), f, "exec") for f in sys.argv[1:]]' \
  scripts/export_deim.py scripts/compare.py
printf 'check: ok\n'
