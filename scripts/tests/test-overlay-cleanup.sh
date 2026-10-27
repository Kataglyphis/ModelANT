#!/usr/bin/env bash
# Exercises both EXIT traps; it does not run the model export.
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/antfrastructure.sh
source "${here}/../lib/antfrastructure.sh"
harness="$(antfrastructure_path linux/scripts/lib/test-harness.sh)" || exit 1
# shellcheck source=third_party/ANTfrastructure/linux/scripts/lib/test-harness.sh
source "${harness}"
helper="$(antfrastructure_path linux/scripts/01-core/uv-image-overlay.sh)" || exit 1
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
mkdir -p "${temp}/fake-bin" "${temp}/target"
printf '#!/usr/bin/env bash\nexit 7\n' > "${temp}/fake-bin/rm"
chmod +x "${temp}/fake-bin/rm"
run_trap() {
  local line="$1" prior="$2"
  env PATH="${temp}/fake-bin:${PATH}" bash -c 'source "$1"; venv="$2"; work="$2"; eval "$3"; exit "$4"' _ "${helper}" "${temp}/target" "${line}" "${prior}"
}
for script in "${here}/../build.sh" "${here}/check-image-lock.sh"; do
  line="$(sed -n "/^trap 'uv_image_overlay_exit /p" "${script}")"
  t_case "${script} keeps its actual EXIT trap and forwards both statuses"
  t_assert_contains "${line}" 'uv_image_overlay_exit' 'the cleanup trap exists'
  [ -n "${line}" ] || continue
  t_assert_rc 1 run_trap "${line}" 0
  t_assert_rc 42 run_trap "${line}" 42
done
t_summary
