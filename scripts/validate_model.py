# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""Check an existing model against real positive and blank reference inputs."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from compare import prepare
from inference_contract import validate_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(args.model, options, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    if inp.type != "tensor(float)" or inp.name != "images" or inp.shape[:2] != [1, 3]:
        raise ValueError(f"unsupported model input: {inp.name} {inp.type} {inp.shape}")
    mode = session.get_modelmeta().custom_metadata_map["kataglyphis.preprocess"]
    fixture = Path(__file__).resolve().parents[1] / "contracts/fixtures/two_cats.jpg"
    with Image.open(fixture) as image:
        prepared, _ = prepare(image, mode, inp.shape[2])
        positive = (np.asarray(prepared, dtype=np.float32) / 255).transpose(2, 0, 1)[None]
    for name, pixels in (("two_cats", positive), ("blank", np.zeros_like(positive))):
        rows = session.run(None, {"images": pixels})[0]
        validate_outputs(rows, rows, require_positive=name == "two_cats")
        cats = sum(row[4] >= 0.25 and int(row[5]) == 15 for row in rows[0])
        if name == "two_cats" and cats < 2:
            raise ValueError(f"positive fixture found only {cats} cats")
        if name == "blank" and any(row[4] >= 0.5 for row in rows[0]):
            raise ValueError("blank fixture produced confident detections")
        print(f"{name}: PASS ({cats} cats above 0.25)")


if __name__ == "__main__":
    main()
