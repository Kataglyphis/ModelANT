# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""Which model licenses ModelANT accepts, and what a community license must record."""

PERMISSIVE = frozenset({"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"})
# Approved by the owner on 2026-10-30; every release and consumer carries out their obligations.
COMMUNITY = frozenset({"LicenseRef-Llama-Community", "LicenseRef-Gemma"})


def license_problems(name: str, spec: dict) -> list[str]:
    """Why a model.json license is not acceptable; empty when it is."""
    license_id = spec.get("license")
    if license_id in PERMISSIVE:
        return []
    if license_id not in COMMUNITY:
        return [f"{name}: license {license_id!r} is neither permissive nor an approved community license"]
    obligations = spec.get("license_obligations")
    if not isinstance(obligations, list) or not all(isinstance(o, str) and o.strip() for o in obligations):
        return [f"{name}: license_obligations must be a list of non-empty strings"]
    if not obligations:
        return [f"{name}: community license {license_id} needs its license_obligations"]
    return []
