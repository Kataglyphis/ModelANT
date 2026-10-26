#!/usr/bin/env bash
# Model checks run inside the family image to use its pinned Python and native dependencies.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
# shellcheck source=lib/antfrastructure.sh
source "${root}/scripts/lib/antfrastructure.sh"
antfrastructure_source linux/scripts/01-core/python-probe.sh
ci_image_python_require check.sh docs/models.md#building || exit $?
python="${PREFLIGHT_PYTHON}"
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

# compile(), not py_compile: nothing is written into the checkout.
"${python}" -I -c 'import sys; [compile(open(f, encoding="utf-8").read(), f, "exec") for f in sys.argv[1:]]' \
  scripts/export_deim.py scripts/compare.py
"${python}" -I -m unittest discover -s scripts/tests -p 'test_*.py'
printf 'check: ok\n'
