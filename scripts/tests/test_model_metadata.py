# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_metadata import main, model_problems

SPEC = {"name": "m", "license": "Apache-2.0",
        "upstream": {"repository": "https://example.invalid/m", "commit": "a" * 40,
                     "checkpoint": {"sha256": "b" * 64}}}


class ModelMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, spec=SPEC, files=("LICENSE", "NOTICE")):
        d = self.models / spec["name"]
        d.mkdir()
        (d / "model.json").write_text(json.dumps(spec))
        for f in files:
            (d / f).write_text(f)
        return d

    def test_a_complete_permissive_model_passes(self):
        self.assertEqual([], model_problems(self.model()))

    def test_the_repository_models_pass(self):
        repo_models = Path(__file__).resolve().parents[2] / "models"
        for d in sorted(p for p in repo_models.iterdir() if p.is_dir()):
            self.assertEqual([], model_problems(d), d.name)

    def test_a_license_outside_the_policy_fails(self):
        d = self.model({**SPEC, "license": "AGPL-3.0-only"})
        self.assertEqual([f"m: license 'AGPL-3.0-only' is neither permissive nor an approved community license"],
                         model_problems(d))

    def test_a_community_license_needs_obligations(self):
        d = self.model({**SPEC, "license": "LicenseRef-Llama-Community"})
        self.assertEqual(["m: license_obligations must be a list of non-empty strings"], model_problems(d))

    def test_missing_files_and_bad_pins_are_all_reported(self):
        spec = {**SPEC, "upstream": {**SPEC["upstream"], "commit": "main", "checkpoint": {"sha256": "x"}}}
        d = self.model(spec, files=("LICENSE",))
        self.assertEqual([f"{d}: no NOTICE", f"{d}: upstream commit is not a full sha",
                          f"{d}: checkpoint sha256 is not 64 hex digits"], model_problems(d))

    def test_main_fails_on_any_problem_and_summarises_each_model(self):
        self.model()
        self.model({**SPEC, "name": "n", "license": "CC-BY-NC-4.0"})
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(self.models)
        self.assertEqual(1, status)
        self.assertEqual("m: Apache-2.0, https://example.invalid/m\nn: CC-BY-NC-4.0, https://example.invalid/m\n",
                         out.getvalue())
        self.assertIn("n: license 'CC-BY-NC-4.0' is neither", err.getvalue())


if __name__ == "__main__":
    unittest.main()
