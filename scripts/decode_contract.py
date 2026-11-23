# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""Check contracts/inference-v1.json's metadata keys, decode cases, raw-head cases and digest cases.

Expected boxes and digests are data, never computed here; the checks hold each case to its rules.
"""
import hashlib
import json
import math
import pathlib
import sys

LAYOUT = "xyxy_score_class"
METADATA_KEYS = ("kataglyphis.preprocess", "kataglyphis.output")
NON_FINITE = ["NaN", "Infinity", "-Infinity"]
RULES = (1, 2, 3, 4, 5)
DECODE_KEYS = {"layout", "rules", "non_finite", "order", "score_tolerance", "coordinate_tolerance", "cases"}
CASE_KEYS = {"name", "rule", "width", "height", "input_width", "input_height", "mode", "mapping",
             "score_threshold", "rows", "expected"}
MAPPING_KEYS = {"content_width", "content_height", "pad_x", "pad_y", "scale_x", "scale_y"}
EXPECTED_KEYS = {"row", "class", "score", "box"}
# The raw heads D6 keeps (hub CON288 (e)): the values before a candidate's class scores, by layout.
RAW_LAYOUTS = {"cxcywh_objectness_scores": 5, "cxcywh_scores_by_anchor": 4}
RAW_KEYS = {"layouts", "rules", "iou_threshold", "order", "score_tolerance", "coordinate_tolerance", "cases"}
RAW_CASE_KEYS = {"name", "rule", "layout", "width", "height", "input_width", "input_height", "mode", "mapping",
                 "score_threshold", "output", "expected"}
RAW_EXPECTED_KEYS = {"candidate", "class", "score", "box"}
DIGEST_KEYS = {"frame", "tensor", "cases"}
DIGEST_CASE_KEYS = {"name", "pixel_format", "width", "height", "row_stride_bytes", "plane_offset_bytes",
                    "input_width", "input_height", "mode", "mapping", "expected_sha256", "samples"}
# The family's raw-head suppression merges boxes above this overlap; a kept pair must exceed it to prove none ran.
SUPPRESSION_IOU = 0.45
# A published case stays as it is under one schema_version (AGENTS.md); a new case records its digest here.
PUBLISHED = {
    "stretch_rows_keep_row_order": "4b57b0b8deb039f2ce8a91fbde1bf803e35e6ab21957e088f7043013dbdc1feb",
    "score_finite_and_at_threshold": "f11409334caee2dfa31fe591ab366b0f06b9b7974cdce5c27b72533bb0c408e5",
    "class_in_range_cut_to_integer": "8a3b97b066645e7e526d8c3d4172595a0fe716a5b0aa6d4f39a03377d4d3e9b9",
    "corners_finite_and_ordered": "ee514df915617ef07ebf9185ca15296b3dcb6bae4de578257dd9dfbbdc36a315",
    "overlapping_boxes_all_stay": "96f800fdf6a6562167c7450745e803b905ccaa5780b429fd3c4c32144d25cca9",
    "box_cut_at_each_frame_edge": "1c7d09acd118dd03b02e584fa0ea11dca26106126887e7a4ea6976a31b650729",
    "letterbox_top_padding": "24e86cf406b4e672c3a69c892dc5eb2b863796bf2cbdd8a638e74acdbc0a9bd7",
    "letterbox_left_padding": "6bf9b5226693d879f56567ea6b2b24ec6578f93f4ae4744079e649f7278fb95a",
    "letterbox_box_in_padding_cut": "ac9b0023364b588163d2511034ca03f652923ee1e6a83b06021c597ef9e9eefb",
}
# The raw-head and full-size digest cases of hub CON288 (e), published as the decode cases are.
PUBLISHED_RAW = {
    "objectness_times_the_best_class": "5a8622bdb61c45516c4481e2c434dc1281ee54286376569f9f2b6b8d452a5633",
    "same_class_overlap_above_0_45_keeps_the_stronger": "f551bb6a06bf4e3944c9719176014a033168e123d35df3fccc6c22170b884524",
    "same_class_overlap_at_most_0_45_stays": "97726463b618d3de0aa65e6475a6024ce5256ee71308b108aa236fd67cedb08a",
    "different_classes_never_suppress": "ad1fa88636aa7f92f4cbe7e298b195d5ae36c797560695079a03f7c125d8fb81",
    "equal_scores_keep_the_lower_candidate": "cc2d8671c0e9643ec450e8e52e3cf560a6792e872d38bc80614017144a60e773",
    "anchor_columns_are_the_candidates": "898eb8eb5397e9458bafca3c2885f322c4d738a9beb553900fa6c7f748ceaa3e",
    "suppression_in_model_input_pixels": "17aec9e17411dbe52a453c2e93638cd2b30c26fdb4d1acb2681e0602a5fb8f1d",
}
PUBLISHED_DIGEST = {
    "full_size_stretch_2304x1296": "f1219a93b59397bb5450946f118ed251c312f42591edfdaa0461c097f557f193",
    "full_size_letterbox_2304x1296": "31a19a0e6e35c50185f466e20373ebedabd4e1fa80f6c4167a66f72b3caacbdd",
    "odd_width_rows_padded_to_4_bytes": "aba5d072bdd990f6e339e74c327aa03339472070efd3bb2bc621cb183e5c2cc3",
    "rgba_with_a_plane_offset": "61958f621849e973340102f43a3e0338b926718f4d5701be69d51536dc437cd0",
}


def load(path: pathlib.Path) -> dict:
    """The contract as strict JSON: Python alone would accept the bare NaN that serde_json and nlohmann refuse."""
    def refuse(token: str):
        raise ValueError(f"{path}: bare {token} is not JSON; a row spells it as a string")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def _number(value) -> bool:
    """A finite JSON number; Python's bool is an int and is none."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value) -> bool:
    """A JSON integer of at least zero."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _names(values) -> bool:
    """A non-empty list of distinct, non-empty strings."""
    return (isinstance(values, list) and bool(values) and all(isinstance(value, str) and value for value in values)
            and len(set(values)) == len(values))


def metadata_problems(spec: dict, models: dict[str, dict]) -> list[str]:
    """Why the listed metadata keys, or a model's values for them, do not fit; empty when they do."""
    listed = spec.get("onnx_metadata")
    if not isinstance(listed, dict) or any(key not in listed for key in METADATA_KEYS):
        return [f"onnx_metadata must list {' and '.join(METADATA_KEYS)}"]
    bad = [f"onnx_metadata {key}: the allowed values must be distinct non-empty strings"
           for key, values in listed.items() if not _names(values)]
    if bad:
        return bad
    if LAYOUT not in listed["kataglyphis.output"]:
        bad.append(f"onnx_metadata kataglyphis.output must allow {LAYOUT}, the layout the decode cases are for")
    for name, model in sorted(models.items()):
        if "onnx" not in model:
            continue
        written = model["onnx"].get("metadata", {})
        bad += [f"{name}: {key} {written.get(key)!r} is not one of {', '.join(listed[key])}"
                for key in METADATA_KEYS if written.get(key) not in listed[key]]
    return bad


def _mapping_form_problem(case: dict) -> str | None:
    """Why the frame, input and mapping fields cannot be compared at all; None when they can."""
    sizes = [case[key] for key in ("width", "height", "input_width", "input_height")]
    mapping = case["mapping"]
    if not all(_count(size) and size for size in sizes) or case["mode"] not in ("stretch", "letterbox"):
        return "width, height, input_width and input_height must be positive integers, mode stretch or letterbox"
    if not isinstance(mapping, dict) or set(mapping) != MAPPING_KEYS:
        return f"mapping must hold exactly {', '.join(sorted(MAPPING_KEYS))}"
    whole = [mapping[key] for key in ("content_width", "content_height", "pad_x", "pad_y")]
    if not all(map(_count, whole)) or 0 in whole[:2]:
        return "mapping: content sizes must be positive integers and paddings integers of at least zero"
    if not all(_number(mapping[key]) and mapping[key] > 0 for key in ("scale_x", "scale_y")):
        return "mapping: scale_x and scale_y must be numbers above zero"
    return None


def mapping_problems(case: dict) -> list[str]:
    """Why a case's frame, input and mapping are not one preprocessing result; empty when they are."""
    form = _mapping_form_problem(case)
    if form:
        return [form]
    mapping = case["mapping"]
    frame, room = (case["width"], case["height"]), (case["input_width"], case["input_height"])
    content = (mapping["content_width"], mapping["content_height"])
    pads = (mapping["pad_x"], mapping["pad_y"])
    scales = (mapping["scale_x"], mapping["scale_y"])
    bad = []
    if any(pad + size > limit for pad, size, limit in zip(pads, content, room)):
        bad.append("mapping: padding plus content does not fit the input")
    if not all(math.isclose(scale * side, size) for scale, side, size in zip(scales, frame, content)):
        bad.append("mapping: a scale times the frame's size is not the content size")
    if case["mode"] == "stretch" and (pads != (0, 0) or content != room):
        bad.append("mapping: a stretch fills the input and has no padding")
    return bad


def rows_problems(rows) -> list[str]:
    """Why the raw rows are not six portable values each; empty when they are."""
    if not isinstance(rows, list) or not rows:
        return ["rows must be a non-empty list"]
    bad = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 6:
            bad.append(f"row {index}: must hold x1, y1, x2, y2, score and class")
        elif not all(_number(value) or value in NON_FINITE for value in row):
            bad.append(f"row {index}: a value is neither a finite number nor one of {', '.join(NON_FINITE)}")
    return bad


def _score_problems(entry: dict, row: list, threshold: float) -> list[str]:
    """Why an expected score breaks rule 1 or is not copied from its row."""
    bad = []
    if not _number(entry["score"]) or entry["score"] < threshold:
        bad.append("rule 1: the score is not a finite number of at least score_threshold")
    if entry["score"] != row[4]:
        bad.append("the score is not the row's score")
    return bad


def _class_problems(entry: dict, row: list, class_range: list) -> list[str]:
    """Why an expected class breaks rule 2 or is not its row's class."""
    bad = []
    if not _count(entry["class"]) or not class_range[0] <= entry["class"] <= class_range[1]:
        bad.append(f"rule 2: the class is no integer from {class_range[0]} to {class_range[1]}")
    if not _number(row[5]) or math.trunc(row[5]) != entry["class"]:
        bad.append("rule 2: the class is not the row's class cut to an integer")
    return bad


def _box_problems(entry: dict, row: list, case: dict) -> list[str]:
    """Why an expected box breaks rule 3 or rule 5, or is not in whole pixels."""
    box = entry["box"]
    if not isinstance(box, list) or len(box) != 4 or not all(_number(v) and v == math.trunc(v) for v in box):
        return ["the box is not four whole pixel numbers"]
    bad = []
    if not all(map(_number, row[:4])) or row[2] < row[0] or row[3] < row[1]:
        bad.append("rule 3: the row's corners are not finite and in order")
    if box[2] < box[0] or box[3] < box[1]:
        bad.append("rule 3: the box's corners are out of order")
    if min(box) < 0 or max(box[0], box[2]) > case["width"] or max(box[1], box[3]) > case["height"]:
        bad.append("rule 5: the box reaches past the frame")
    return bad


def expected_problems(case: dict, class_range: list) -> list[str]:
    """Why an expected detection breaks a rule, is not its row's, or is out of order; empty when all hold."""
    rows, expected = case["rows"], case["expected"]
    if not isinstance(expected, list):
        return ["expected must be a list"]
    bad, last = [], -1
    for entry in expected:
        if not isinstance(entry, dict) or set(entry) != EXPECTED_KEYS:
            bad.append(f"an expected detection must hold exactly {', '.join(sorted(EXPECTED_KEYS))}")
            continue
        row = entry["row"]
        if not _count(row) or row >= len(rows):
            bad.append(f"expected row {row!r} is no row of the case")
            continue
        if row <= last:
            bad.append(f"expected row {row} is out of order: detections come in the order of their rows")
        last = row
        problems = (_score_problems(entry, rows[row], case["score_threshold"])
                    + _class_problems(entry, rows[row], class_range) + _box_problems(entry, rows[row], case))
        bad += [f"expected row {row}: {why}" for why in problems]
    return bad


def _iou(a: list, b: list) -> float:
    """Intersection over union of two x1, y1, x2, y2 boxes."""
    overlap = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def overlap_problems(case: dict) -> list[str]:
    """Why a rule 4 case would pass a decoder that still suppressed overlaps; empty when it would not."""
    if case["rule"] != 4:
        return []
    kept, bad = case["expected"], []
    if len(kept) != len(case["rows"]):
        bad.append("rule 4: every row of a suppression case must be expected")
    pairs = [(a, b) for index, a in enumerate(kept) for b in kept[index + 1:] if a["class"] == b["class"]]
    if not any(_iou(a["box"], b["box"]) > SUPPRESSION_IOU for a, b in pairs):
        bad.append(f"rule 4: no two expected boxes of one class overlap above IoU {SUPPRESSION_IOU}")
    return bad


def _rule_and_threshold_problems(case: dict) -> list[str]:
    """Why a case's rule or score_threshold is no rule number or no usable threshold."""
    bad = []
    if not _count(case["rule"]) or case["rule"] not in RULES:
        bad.append("rule must be 1, 2, 3, 4 or 5")
    if not _number(case["score_threshold"]) or not 0 < case["score_threshold"] <= 1:
        bad.append("score_threshold must be a number above 0 and at most 1")
    return bad


def case_problems(case, class_range: list) -> list[str]:
    """Why one decode case is malformed or does not show its rule; empty when it is sound."""
    if not isinstance(case, dict) or set(case) != CASE_KEYS:
        return [f"a decode case must hold exactly {', '.join(sorted(CASE_KEYS))}"]
    if not isinstance(case["name"], str) or not case["name"]:
        return ["a decode case needs a name"]
    bad = mapping_problems(case) + rows_problems(case["rows"]) + _rule_and_threshold_problems(case)
    bad = bad or expected_problems(case, class_range)
    bad = bad or overlap_problems(case)
    return [f"{case['name']}: {why}" for why in bad]


def _settings_problems(decode: dict) -> list[str]:
    """Why the section's own keys are not the ones the cases are written for."""
    bad = []
    if decode["layout"] != LAYOUT:
        bad.append(f"decode.layout must be {LAYOUT}: the rules and rows are that layout's")
    if decode["non_finite"] != NON_FINITE:
        bad.append(f"decode.non_finite must be {', '.join(NON_FINITE)}")
    if decode["order"] != "row":
        bad.append("decode.order must be row")
    return bad + _tolerance_and_rules_problems(decode, "decode")


def _tolerance_and_rules_problems(section: dict, label: str) -> list[str]:
    """Why a decode section's tolerances or its five numbered rules do not fit."""
    bad = []
    for key, limit in (("score_tolerance", 0.01), ("coordinate_tolerance", 0.5)):
        if not _number(section[key]) or not 0 < section[key] < limit:
            bad.append(f"{label}.{key} must be a number above 0 and under {limit}")
    rules = section["rules"]
    if not isinstance(rules, list) or [str(rule)[:3] for rule in rules] != [f"{rule}. " for rule in RULES]:
        bad.append(f"{label}.rules must be the five rules, each starting with its number")
    return bad


def decode_problems(spec: dict) -> list[str]:
    """Why the decode section is malformed; empty when every case is sound and every rule has one."""
    decode = spec.get("decode")
    if not isinstance(decode, dict) or set(decode) != DECODE_KEYS:
        return [f"decode must hold exactly {', '.join(sorted(DECODE_KEYS))}"]
    cases = decode["cases"]
    if not isinstance(cases, list) or not cases:
        return ["decode.cases must be a non-empty list"]
    bad = _settings_problems(decode)
    for case in cases:
        bad += case_problems(case, spec["output"]["class_range"])
    if bad:
        return bad
    names = [case["name"] for case in cases] + [case["name"] for case in spec["cases"]]
    bad += [f"case name {name} is used twice" for name in sorted(set(names)) if names.count(name) > 1]
    bad += [f"rule {rule} has no decode case" for rule in RULES if all(case["rule"] != rule for case in cases)]
    return bad


def raw_candidates(case: dict) -> list[list]:
    """One list per candidate: the output's rows, or its columns for cxcywh_scores_by_anchor."""
    if case["layout"] == "cxcywh_scores_by_anchor":
        return [list(column) for column in zip(*case["output"])]
    return case["output"]


def raw_scores(case: dict) -> list[tuple[int, float, list]]:
    """Each candidate's class, score and corners in model-input pixels, by raw rules 1 and 3."""
    lead = RAW_LAYOUTS[case["layout"]]
    scored = []
    for values in raw_candidates(case):
        classes = values[lead:]
        best = max(classes)
        score = values[4] * best if lead == 5 else best
        cx, cy, w, h = values[:4]
        scored.append((classes.index(best), score, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return scored


def _raw_output_problems(case: dict) -> list[str]:
    """Why the output is not a matrix of finite numbers in its layout's shape; empty when it is."""
    output, lead = case["output"], RAW_LAYOUTS.get(case["layout"])
    if lead is None:
        return [f"layout must be one of {', '.join(RAW_LAYOUTS)}"]
    if not isinstance(output, list) or not output or not all(isinstance(row, list) and row for row in output):
        return ["output must be a non-empty list of non-empty lists"]
    if len({len(row) for row in output}) != 1 or not all(_number(value) for row in output for value in row):
        return ["output must be a matrix of finite numbers"]
    if len(raw_candidates(case)[0]) <= lead:
        return [f"a {case['layout']} candidate holds more than {lead} values, a class score among them"]
    return []


def _whole_box_in_frame(box, case: dict) -> bool:
    """Four whole pixel numbers in order, inside the frame."""
    if not isinstance(box, list) or len(box) != 4 or not all(_number(v) and v == math.trunc(v) for v in box):
        return False
    return box[0] <= box[2] <= case["width"] and box[1] <= box[3] <= case["height"] and min(box) >= 0


def _raw_entry_problems(entry: dict, scored: list, case: dict) -> list[str]:
    """Why one expected detection is not its candidate's class and score, or its box no whole box in the frame."""
    index = entry["candidate"]
    if not _count(index) or index >= len(scored):
        return [f"expected candidate {index!r} is no candidate of the case"]
    cls, score, _ = scored[index]
    bad = []
    if entry["class"] != cls:
        bad.append(f"candidate {index}: rule 1: the class is not the first class column holding the best score")
    if not _number(entry["score"]) or not math.isclose(entry["score"], score, abs_tol=1e-9):
        bad.append(f"candidate {index}: rule 1: the score is not the candidate's score")
    if score < case["score_threshold"]:
        bad.append(f"candidate {index}: rule 2: the score is under score_threshold")
    if not _whole_box_in_frame(entry["box"], case):
        bad.append(f"candidate {index}: rule 5: the box is not four whole pixels in order inside the frame")
    return bad


def _ranked_before(first: int, second: int, scored: list) -> bool:
    """Whether candidate first comes before second: a higher score, ties to the lower candidate."""
    return (-scored[first][1], first) < (-scored[second][1], second)


def _suppressed_by_kept(index: int, kept: list, scored: list) -> bool:
    """Whether a kept candidate ranked before index holds its class and overlaps it above SUPPRESSION_IOU."""
    cls, _, box = scored[index]
    return any(scored[k][0] == cls and _ranked_before(k, index, scored) and _iou(scored[k][2], box) > SUPPRESSION_IOU
               for k in kept)


def _raw_suppression_problems(case: dict, scored: list) -> list[str]:
    """Why the expected set is not what rule 4 keeps, judged in model-input pixels; empty when it is."""
    kept = [entry["candidate"] for entry in case["expected"]]
    bad = []
    for index, (_, score, _) in enumerate(scored):
        dropped = index not in kept and score >= case["score_threshold"]
        if dropped and not _suppressed_by_kept(index, kept, scored):
            bad.append(f"candidate {index}: rule 4: it is confident and not expected, yet no kept candidate of its class "
                       f"ranked before it overlaps it above IoU {SUPPRESSION_IOU}")
    bad += [f"candidate {k}: rule 4: it is expected, yet a kept candidate of its class ranked before it overlaps it "
            f"above IoU {SUPPRESSION_IOU}" for k in kept if _suppressed_by_kept(k, kept, scored)]
    return bad


def _raw_expected_problems(case: dict) -> list[str]:
    """Why the expected detections break a raw rule or come out of order; empty when they hold."""
    entries = case["expected"]
    if not isinstance(entries, list) or not all(isinstance(e, dict) and set(e) == RAW_EXPECTED_KEYS for e in entries):
        return [f"expected must be a list of detections, each holding exactly {', '.join(sorted(RAW_EXPECTED_KEYS))}"]
    scored = raw_scores(case)
    bad = [why for entry in entries for why in _raw_entry_problems(entry, scored, case)]
    if bad:
        return bad
    candidates = [entry["candidate"] for entry in entries]
    if candidates != sorted(candidates, key=lambda index: (-scored[index][1], index)):
        bad.append("rule 4: the detections must come in descending score, ties to the lower candidate")
    return bad + _raw_suppression_problems(case, scored)


def raw_case_problems(case) -> list[str]:
    """Why one raw-head case is malformed or breaks a raw rule; empty when it is sound."""
    if not isinstance(case, dict) or set(case) != RAW_CASE_KEYS:
        return [f"a raw case must hold exactly {', '.join(sorted(RAW_CASE_KEYS))}"]
    if not isinstance(case["name"], str) or not case["name"]:
        return ["a raw case needs a name"]
    bad = mapping_problems(case) + _raw_output_problems(case) + _rule_and_threshold_problems(case)
    bad = bad or _raw_expected_problems(case)
    return [f"{case['name']}: {why}" for why in bad]


def raw_decode_problems(spec: dict) -> list[str]:
    """Why the raw_decode section is malformed; empty when every case is sound and every layout has one."""
    section = spec.get("raw_decode")
    if not isinstance(section, dict) or set(section) != RAW_KEYS:
        return [f"raw_decode must hold exactly {', '.join(sorted(RAW_KEYS))}"]
    cases = section["cases"]
    if not isinstance(cases, list) or not cases:
        return ["raw_decode.cases must be a non-empty list"]
    bad = _tolerance_and_rules_problems(section, "raw_decode")
    if not isinstance(section["layouts"], dict) or set(section["layouts"]) != set(RAW_LAYOUTS):
        bad.append(f"raw_decode.layouts must describe exactly {', '.join(RAW_LAYOUTS)}")
    if section["iou_threshold"] != SUPPRESSION_IOU or section["order"] != "score":
        bad.append(f"raw_decode.iou_threshold must be {SUPPRESSION_IOU} and raw_decode.order score")
    for case in cases:
        bad += raw_case_problems(case)
    if bad:
        return bad
    return [f"layout {layout} has no raw case" for layout in RAW_LAYOUTS if all(c["layout"] != layout for c in cases)]


def _digest_frame_problems(case: dict) -> list[str]:
    """Why a full-size case's pixel format, offset and row stride cannot build its frame."""
    channels = len(case["pixel_format"]) if case["pixel_format"] in ("RGB", "RGBA") else 0
    if not channels or not _count(case["plane_offset_bytes"]) or not _count(case["row_stride_bytes"]):
        return ["pixel_format must be RGB or RGBA, and the offset and the row stride integers of at least zero"]
    if case["row_stride_bytes"] < case["width"] * channels:
        return ["row_stride_bytes is shorter than a row of pixels"]
    return []


def _digest_expected_problems(case: dict) -> list[str]:
    """Why a full-size case's expected digest or samples cannot be compared with a tensor."""
    bad = []
    digest = case["expected_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
        bad.append("expected_sha256 must be 64 lowercase hex digits")
    samples = case["samples"]
    if not isinstance(samples, list) or not samples or not all(
            isinstance(s, list) and len(s) == 4 and all(map(_count, s[:3])) and _number(s[3]) for s in samples):
        bad.append("samples must list channel, y, x and the value there")
    return bad


def _digest_case_problems(case) -> list[str]:
    """Why one full-size case cannot be replayed: its frame, its mapping or its expected digest and samples."""
    if not isinstance(case, dict) or set(case) != DIGEST_CASE_KEYS:
        return [f"a digest case must hold exactly {', '.join(sorted(DIGEST_CASE_KEYS))}"]
    bad = mapping_problems(case) + _digest_frame_problems(case) + _digest_expected_problems(case)
    return [f"{case.get('name')}: {why}" for why in bad]


def digest_problems(spec: dict) -> list[str]:
    """Why the digest_cases section is malformed; empty when every full-size case can be replayed."""
    section = spec.get("digest_cases")
    if not isinstance(section, dict) or set(section) != DIGEST_KEYS:
        return [f"digest_cases must hold exactly {', '.join(sorted(DIGEST_KEYS))}"]
    if not all(isinstance(section[key], str) and section[key] for key in ("frame", "tensor")):
        return ["digest_cases.frame and digest_cases.tensor must say how the frame and the tensor are built"]
    cases = section["cases"]
    if not isinstance(cases, list) or not cases:
        return ["digest_cases.cases must be a non-empty list"]
    return [why for case in cases for why in _digest_case_problems(case)]


def case_digest(case: dict) -> str:
    """SHA-256 of a case in one canonical spelling, so reformatting the file changes no digest."""
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _published(cases: list, published: dict, label: str, listed_in: str) -> list[str]:
    """Why these cases are not the published ones plus recorded additions; empty when they are."""
    found = {case["name"]: case_digest(case) for case in cases}
    bad = [f"{label} {name} is missing: a published case stays" for name in published if name not in found]
    for name, digest in found.items():
        if name not in published:
            bad.append(f"{label} {name} is not recorded: add its digest {digest} to {listed_in}")
        elif digest != published[name]:
            bad.append(f"{label} {name} changed (digest {digest}): a published case changes only with schema_version")
    return bad


def published_problems(spec: dict) -> list[str]:
    """Why the decode, raw-head and digest cases are not the published ones plus recorded additions."""
    return (_published(spec["decode"]["cases"], PUBLISHED, "decode case", "PUBLISHED")
            + _published(spec["raw_decode"]["cases"], PUBLISHED_RAW, "raw case", "PUBLISHED_RAW")
            + _published(spec["digest_cases"]["cases"], PUBLISHED_DIGEST, "digest case", "PUBLISHED_DIGEST"))


def contract_problems(spec: dict, models: dict[str, dict]) -> list[str]:
    """Every problem of the metadata keys and the decode section; empty when the contract is sound."""
    bad = metadata_problems(spec, models) + decode_problems(spec) + raw_decode_problems(spec) + digest_problems(spec)
    # The digests of malformed cases would only repeat their cause.
    return bad or published_problems(spec)


def model_specs(models: pathlib.Path) -> dict[str, dict]:
    """Every models/<name>/model.json by its directory name."""
    paths = sorted(models.glob("*/model.json"))
    return {path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in paths}


def main(contract: pathlib.Path, models: pathlib.Path) -> int:
    """Print what the decode section holds; 1 with every problem on stderr."""
    spec = load(contract)
    bad = contract_problems(spec, model_specs(models))
    for line in bad:
        print(f"{contract.name}: {line}", file=sys.stderr)
    if not bad:
        cases = spec["decode"]["cases"]
        print(f"{contract.name}: {len(cases)} decode cases, {sum(len(case['rows']) for case in cases)} rows, "
              f"layout {spec['decode']['layout']}; {len(spec['raw_decode']['cases'])} raw-head cases, "
              f"{len(spec['digest_cases']['cases'])} full-size digest cases")
    return 1 if bad else 0


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = [pathlib.Path(arg) for arg in sys.argv[1:3]]
    sys.exit(main(*(paths + [root / "contracts/inference-v1.json", root / "models"][len(paths):])))
