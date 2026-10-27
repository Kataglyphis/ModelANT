#!/usr/bin/env bash
# Verify declared project and fixture notices; this does not discover other dependency obligations.
set -euo pipefail

out="${1:?usage: pack-contract.sh <archive>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/antfrastructure.sh
source "${root}/scripts/lib/antfrastructure.sh"
verifier="$(antfrastructure_path linux/scripts/verify_package_license.py)"
temporary="$(mktemp "${out}.tmp.XXXXXXXXXX")"
trap 'rm -f "${temporary}"' EXIT

tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "${root}" \
  -cf - LICENSE NOTICE contracts | gzip -n > "${temporary}"
python3 "${verifier}" "${temporary}" \
  --require "LICENSE=${root}/LICENSE" --require "NOTICE=${root}/NOTICE" \
  --require "contracts/fixtures/LICENSE=${root}/contracts/fixtures/LICENSE" \
  --require "contracts/fixtures/NOTICE=${root}/contracts/fixtures/NOTICE"
mv -f "${temporary}" "${out}"
