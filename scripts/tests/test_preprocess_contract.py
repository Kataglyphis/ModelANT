# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import prepare

CONTRACT = Path(__file__).resolve().parents[2] / "contracts/inference-v1.json"


class PortablePreprocessTests(unittest.TestCase):
    def assert_content_maps_to_the_frame(self, case, back):
        mapping = case["mapping"]
        box = back(mapping["pad_x"], mapping["pad_y"],
                   mapping["pad_x"] + mapping["content_width"],
                   mapping["pad_y"] + mapping["content_height"])
        np.testing.assert_allclose(box, [0, 0, case["width"], case["height"]], atol=1e-7)

    def test_decode_case_mappings_are_what_the_preprocessing_returns(self):
        spec = json.loads(CONTRACT.read_text(encoding="utf-8"))
        for case in spec["decode"]["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(case["input_width"], case["input_height"])
                frame = Image.new("RGB", (case["width"], case["height"]))
                self.assert_content_maps_to_the_frame(case, prepare(frame, case["mode"], case["input_width"])[1])

    def test_all_portable_cases_match_production_comparison(self):
        spec = json.loads(CONTRACT.read_text(encoding="utf-8"))
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
                self.assert_content_maps_to_the_frame(case, back)

    def test_full_size_frames_match_their_digest_and_samples(self):
        # Hub CON288 (e): the frame from digest_cases.frame, the tensor's bytes from digest_cases.tensor.
        spec = json.loads(CONTRACT.read_text(encoding="utf-8"))
        for case in spec["digest_cases"]["cases"]:
            with self.subTest(case=case["name"]):
                fmt, width, height = case["pixel_format"], case["width"], case["height"]
                stride, offset = case["row_stride_bytes"], case["plane_offset_bytes"]
                raw = np.full(offset + stride * height, 171, dtype=np.uint8)
                ys, xs, cs = np.meshgrid(np.arange(height), np.arange(width), np.arange(len(fmt)), indexing="ij")
                pixels = ((7 * xs + 13 * ys + 61 * cs) % 256).astype(np.uint8).reshape(height, -1)
                for y in range(height):
                    raw[offset + y * stride:offset + y * stride + pixels.shape[1]] = pixels[y]
                image = Image.frombytes(fmt, (width, height), raw.tobytes()[offset:], "raw", fmt, stride, 1)
                prepared, back = prepare(image, case["mode"], case["input_width"])
                tensor = (np.asarray(prepared, dtype=np.float32) / np.float32(255)).transpose(2, 0, 1)
                for channel, y, x, value in case["samples"]:
                    self.assertEqual(float(tensor[channel, y, x]), value, (channel, y, x))
                self.assertEqual(hashlib.sha256(tensor.astype("<f4").tobytes()).hexdigest(), case["expected_sha256"])
                self.assert_content_maps_to_the_frame(case, back)


if __name__ == "__main__":
    unittest.main()
