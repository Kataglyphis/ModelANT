# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from license_policy import license_problems

LLAMA = {"license": "LicenseRef-Llama-Community", "license_obligations": ["display Built with Llama"]}


class LicensePolicyTests(unittest.TestCase):
    def test_permissive_licenses_need_nothing_else(self):
        for license_id in ("Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"):
            self.assertEqual([], license_problems("m", {"license": license_id}))

    def test_an_unlisted_license_is_refused(self):
        for license_id in ("AGPL-3.0-only", "CC-BY-NC-4.0", "", None):
            self.assertEqual(
                [f"m: license {license_id!r} is neither permissive nor an approved community license"],
                license_problems("m", {"license": license_id}))

    def test_a_community_license_with_obligations_passes(self):
        self.assertEqual([], license_problems("m", LLAMA))
        self.assertEqual([], license_problems("m", {**LLAMA, "license": "LicenseRef-Gemma"}))

    def test_a_community_license_without_obligations_is_refused(self):
        bare = {"license": "LicenseRef-Gemma"}
        self.assertEqual(["m: license_obligations must be a list of non-empty strings"],
                         license_problems("m", bare))
        self.assertEqual(["m: community license LicenseRef-Gemma needs its license_obligations"],
                         license_problems("m", {**bare, "license_obligations": []}))

    def test_blank_or_non_string_obligations_are_refused(self):
        for obligations in (["  "], [1], "display Built with Llama"):
            self.assertEqual(["m: license_obligations must be a list of non-empty strings"],
                             license_problems("m", {**LLAMA, "license_obligations": obligations}))


if __name__ == "__main__":
    unittest.main()
