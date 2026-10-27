#!/usr/bin/env bash
# Checks the actual contract archive and failed verification; it does not grade model licenses.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/antfrastructure.sh
source "${root}/scripts/lib/antfrastructure.sh"
# shellcheck disable=SC1090
source "$(antfrastructure_path linux/scripts/tests/test-harness.sh)"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
archive_matches() {
  tar -xOzf "$1" "$2" | cmp - "${root}/$2"
}

t_case 'contract archives contain all declared project and fixture texts and are reproducible'
t_assert_ok bash "${root}/scripts/pack-contract.sh" "${work}/one.tar.gz"
t_assert_ok bash "${root}/scripts/pack-contract.sh" "${work}/two.tar.gz"
t_assert_ok cmp "${work}/one.tar.gz" "${work}/two.tar.gz"
for relative in LICENSE NOTICE contracts/fixtures/LICENSE contracts/fixtures/NOTICE; do
  t_assert_ok archive_matches "${work}/one.tar.gz" "${relative}"
done

t_case 'verification rejects an archive missing the project license without replacing a prior output'
mkdir "${work}/bin"
real_tar="$(command -v tar)"
cat > "${work}/bin/tar" <<'SH'
#!/usr/bin/env bash
args=()
for arg in "$@"; do
  [[ "$arg" == LICENSE ]] || args+=("$arg")
done
exec "${REAL_TAR}" "${args[@]}"
SH
chmod +x "${work}/bin/tar"
t_assert_fails env PATH="${work}/bin:${PATH}" REAL_TAR="${real_tar}" \
  bash "${root}/scripts/pack-contract.sh" "${work}/one.tar.gz"
t_assert_ok cmp "${work}/one.tar.gz" "${work}/two.tar.gz"
t_summary
