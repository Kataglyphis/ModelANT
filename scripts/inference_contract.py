# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""Validate detector outputs with separate coordinate, score and class contracts."""
import math


def _rows(value, name):
    if len(value) != 1 or len(value[0]) != 300:
        raise ValueError(f"{name}: expected shape [1, 300, 6]")
    rows = value[0]
    for index, row in enumerate(rows):
        if len(row) != 6 or not all(math.isfinite(float(v)) for v in row):
            raise ValueError(f"{name}[{index}]: expected six finite values")
        score, cls = float(row[4]), float(row[5])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name}[{index}]: confidence outside [0, 1]")
        if cls != int(cls) or not 0 <= cls < 80:
            raise ValueError(f"{name}[{index}]: class must be an integer COCO index")
    return rows


def validate_outputs(expected, actual, *, require_positive=False, score_threshold=0.05,
                     box_atol=1.0, score_atol=1e-4, score_rtol=1e-3):
    """Reject invalid values and mismatched confident detections, including empty positives."""
    reference = _rows(expected, "torch")
    candidate = _rows(actual, "onnxruntime")
    positives = 0
    max_box_error = 0.0
    max_score_error = 0.0
    for index, (want, got) in enumerate(zip(reference, candidate)):
        score_error = abs(float(got[4]) - float(want[4]))
        max_score_error = max(max_score_error, score_error)
        if score_error > score_atol + score_rtol * abs(float(want[4])):
            raise ValueError(f"query {index}: confidence mismatch {got[4]} != {want[4]}")
        if max(float(want[4]), float(got[4])) <= score_threshold:
            continue
        positives += 1
        if int(got[5]) != int(want[5]):
            raise ValueError(f"query {index}: class mismatch {got[5]} != {want[5]}")
        error = max(abs(float(got[j]) - float(want[j])) for j in range(4))
        max_box_error = max(max_box_error, error)
        if error > box_atol:
            raise ValueError(f"query {index}: box error {error} exceeds {box_atol} pixels")
    if require_positive and positives == 0:
        raise ValueError("positive fixture produced no confident detections")
    return {"queries": positives, "max_box_error": max_box_error, "max_score_error": max_score_error}
