# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""two_cats.png is the lossless decode of two_cats.jpg, so every detector reads the same pixels."""
import hashlib
import re
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

FIXTURES = Path(__file__).resolve().parents[2] / "contracts/fixtures"


class LosslessFixtureTests(unittest.TestCase):
    def test_the_png_is_the_jpeg_decoded(self) -> None:
        png = np.asarray(Image.open(FIXTURES / "two_cats.png"))
        jpg = np.asarray(Image.open(FIXTURES / "two_cats.jpg").convert("RGB"))
        self.assertEqual(png.shape, jpg.shape, "the same frame, three channels, no alpha")
        # JPEG decoders differ by a step or two per channel; this one was Pillow's, so it holds exactly today.
        self.assertLessEqual(int(np.abs(png.astype(int) - jpg.astype(int)).max()), 2)

    def test_the_notice_pins_its_digest(self) -> None:
        digest = hashlib.sha256((FIXTURES / "two_cats.png").read_bytes()).hexdigest()
        notice = (FIXTURES / "NOTICE").read_text(encoding="utf-8")
        self.assertIn(digest, re.findall(r"[0-9a-f]{64}", notice), "a re-encode changes the bytes the detectors pin")


if __name__ == "__main__":
    unittest.main()
