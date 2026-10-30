# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_genai

SPEC = json.loads((Path(__file__).resolve().parents[2] / "models" / "smollm2_135m_instruct" / "model.json").read_text())
CASE = {"name": "c", "must_contain": "Paris", "expected_ids": [1, 2]}


class ExportGenaiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.model = self.work / "m-cpu"
        (self.model / "sub").mkdir(parents=True)
        (self.model / "genai_config.json").write_text("{}")
        (self.model / "sub" / "model.onnx").write_bytes(b"onnx")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_upstream_lock_pins_every_file_at_the_revision(self):
        lines = export_genai.upstream_lock(SPEC).splitlines()
        up = SPEC["upstream"]
        self.assertEqual(1 + len(up["files"]), len(lines))
        base = f"{up['repository']}/resolve/{up['commit']}"
        self.assertIn(f"{up['checkpoint']['sha256']}  model.safetensors  {base}/model.safetensors", lines)
        for f in up["files"]:
            self.assertIn(f"{f['sha256']}  {f['file']}  {base}/{f['file']}", lines)

    def test_a_case_holds_only_with_the_text_and_the_recorded_ids(self):
        self.assertEqual([], export_genai.case_problems(CASE, [1, 2], "It is Paris."))
        self.assertEqual(["c: 'Lyon' does not contain 'Paris'"], export_genai.case_problems(CASE, [1, 2], "Lyon"))
        self.assertEqual(["c: produced [1, 3], the contract records [1, 2]"],
                         export_genai.case_problems(CASE, [1, 3], "Paris"))

    def test_an_unrecorded_case_fails_and_prints_what_to_record(self):
        self.assertEqual(["c: no expected_ids recorded; this export produced [7, 8]"],
                         export_genai.case_problems({**CASE, "expected_ids": None}, [7, 8], "Paris"))

    def test_a_model_without_a_case_fails(self):
        self.assertEqual(["m-cpu: no case in text-generation-v1.json"],
                         export_genai.contract_problems(self.model, {"cases": [{"model": "other"}]}))

    def test_every_released_model_directory_has_a_case(self):
        contract = json.loads(export_genai.CONTRACT.read_text())
        named = {c["model"] for c in contract["cases"]}
        self.assertIn(SPEC["genai"]["directory"], named)

    def test_the_archive_is_reproducible_and_has_one_top_directory(self):
        first, second = self.work / "a.tar.xz", self.work / "b.tar.xz"
        export_genai.pack(self.model, first)
        (self.model / "genai_config.json").touch()
        export_genai.pack(self.model, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with tarfile.open(first) as tar:
            members = tar.getmembers()
        self.assertEqual(["m-cpu", "m-cpu/genai_config.json", "m-cpu/sub", "m-cpu/sub/model.onnx"],
                         [m.name for m in members])
        self.assertEqual({(0, 0, 0)}, {(m.mtime, m.uid, m.gid) for m in members})

    @unittest.skipUnless(shutil.which("dpkg-deb"), "dpkg-deb is not installed")
    def test_the_package_installs_the_model_and_its_notices(self):
        notice = self.work / "NOTICE"
        notice.write_text("notice text")
        spec = {**SPEC, "package": {"name": "kataglyphis-model-m", "install_dir": "/usr/share/kataglyphis/models/m-cpu"}}
        deb = export_genai.deb(spec, self.model, [notice], self.work / "stage", self.work)
        self.assertEqual(f"kataglyphis-model-m_{SPEC['release']}_all.deb", deb.name)
        listing = subprocess.run(["dpkg-deb", "--contents", str(deb)], capture_output=True, text=True, check=True).stdout
        self.assertIn("./usr/share/kataglyphis/models/m-cpu/sub/model.onnx", listing)
        self.assertIn("./usr/share/doc/kataglyphis-model-m/copyright", listing)
        fields = subprocess.run(["dpkg-deb", "--field", str(deb)], capture_output=True, text=True, check=True).stdout
        self.assertIn("Architecture: all", fields)
        self.assertIn(f"Version: {SPEC['release']}", fields)


if __name__ == "__main__":
    unittest.main()
