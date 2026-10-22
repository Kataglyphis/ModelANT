# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import prepare


class PortablePreprocessTests(unittest.TestCase):
    def test_all_portable_cases_match_production_comparison(self):
        path = Path(__file__).resolve().parents[2] / "contracts/inference-v1.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(spec["schema_version"], 1)
        self.assertGreaterEqual(len(spec["cases"]), 4)
        for case in spec["cases"]:
            with self.subTest(case=case["name"]):
                pixels = bytes(case["data"])[case["plane_offset_bytes"]:]
                image = Image.frombytes(case["pixel_format"], (case["width"], case["height"]), pixels,
                                        "raw", case["pixel_format"], case["row_stride_bytes"], 1)
                prepared, back = prepare(image, case["mode"], case["input_width"])
                tensor = (np.asarray(prepared, dtype=np.float32) / 255).transpose(2, 0, 1).ravel()
                np.testing.assert_allclose(tensor, case["expected_nchw"], atol=1e-7, rtol=0)
                mapping = case["mapping"]
                box = back(mapping["pad_x"], mapping["pad_y"],
                           mapping["pad_x"] + mapping["content_width"],
                           mapping["pad_y"] + mapping["content_height"])
                np.testing.assert_allclose(box, [0, 0, case["width"], case["height"]], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
