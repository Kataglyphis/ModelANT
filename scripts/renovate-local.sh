#!/usr/bin/env bash
# Keep behavior in the hub driver; see third_party/ANTfrastructure/docs/dependency-updates.md.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/antfrastructure.sh
source "${_SCRIPT_DIR}/lib/antfrastructure.sh"

antfrastructure_exec linux/scripts/renovate-local.sh "$KATAGLYPHIS_REPO_ROOT" "$@"
