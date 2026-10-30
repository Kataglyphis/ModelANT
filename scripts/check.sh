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

"${python}" -I scripts/model_metadata.py models

# compile(), not py_compile: nothing is written into the checkout.
"${python}" -I -c 'import sys; [compile(open(f, encoding="utf-8").read(), f, "exec") for f in sys.argv[1:]]' \
  scripts/export_deim.py scripts/compare.py scripts/license_policy.py scripts/model_metadata.py scripts/export_genai.py
"${python}" -I -m unittest discover -s scripts/tests -p 'test_*.py'
bash scripts/tests/test-overlay-cleanup.sh
bash scripts/tests/check-image-lock.sh
bash scripts/tests/test-contract-package.sh
printf 'check: ok\n'
