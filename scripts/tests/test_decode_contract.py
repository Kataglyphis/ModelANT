# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decode_contract as checks

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/inference-v1.json"
DEIM = {"onnx": {"metadata": {"kataglyphis.preprocess": "stretch", "kataglyphis.output": "xyxy_score_class"}}}
PORTABLE = "a value is neither a finite number nor one of NaN, Infinity, -Infinity"

# Each row: the case, the fields to overwrite, and the one problem the checks must then name.
BROKEN_CASES = (
    ("score_finite_and_at_threshold", [("rows", 4, 4, 0.4), ("expected", 0, "score", 0.4)],
     "expected row 4: rule 1: the score is not a finite number of at least score_threshold"),
    ("score_finite_and_at_threshold", [("expected", 0, "score", 0.6)],
     "expected row 4: the score is not the row's score"),
    ("class_in_range_cut_to_integer", [("expected", 0, {"row": 0, "class": 80, "score": 0.9, "box": [20, 16, 120, 96]})],
     "expected row 0: rule 2: the class is no integer from 0 to 79"),
    ("class_in_range_cut_to_integer", [("expected", 0, "class", 16)],
     "expected row 5: rule 2: the class is not the row's class cut to an integer"),
    ("corners_finite_and_ordered", [("expected", 0, {"row": 0, "class": 15, "score": 0.9, "box": [20, 16, 120, 96]})],
     "expected row 0: rule 3: the row's corners are not finite and in order"),
    ("corners_finite_and_ordered", [("expected", 0, {"row": 4, "class": 15, "score": 0.9, "box": [500, 16, 600, 96]})],
     "expected row 4: rule 3: the row's corners are not finite and in order"),
    ("corners_finite_and_ordered", [("expected", 1, "box", [960, 16, 860, 96])],
     "expected row 7: rule 3: the box's corners are out of order"),
    ("overlapping_boxes_all_stay", [("expected", slice(1, 2), [])],
     "rule 4: every row of a suppression case must be expected"),
    ("overlapping_boxes_all_stay",
     [("rows", slice(1, 4), [[180, 200, 380, 400, 0.6, 15]]),
      ("expected", slice(1, 4), [{"row": 1, "class": 15, "score": 0.6, "box": [360, 320, 760, 640]}])],
     "rule 4: no two expected boxes of one class overlap above IoU 0.45"),
    ("box_cut_at_each_frame_edge", [("expected", 2, "box", [1200, 160, 1300, 320])],
     "expected row 2: rule 5: the box reaches past the frame"),
    ("box_cut_at_each_frame_edge", [("expected", 0, "box", [-20, 160, 100, 320])],
     "expected row 0: rule 5: the box reaches past the frame"),
    ("box_cut_at_each_frame_edge", [("expected", 3, "box", [200, 960, 400, 1032])],
     "expected row 3: rule 5: the box reaches past the frame"),
    ("stretch_rows_keep_row_order", [("expected", 0, "box", [40.5, 80, 240, 480])],
     "expected row 0: the box is not four whole pixel numbers"),
    ("stretch_rows_keep_row_order", [("expected", 0, {"row": 1, "class": 15, "score": 0.9, "box": [200, 320, 600, 640]})],
     "expected row 1 is out of order: detections come in the order of their rows"),
    ("stretch_rows_keep_row_order", [("expected", 1, "row", 2)], "expected row 2 is no row of the case"),
    ("stretch_rows_keep_row_order", [("expected", 1, "note", "x")],
     "an expected detection must hold exactly box, class, row, score"),
    ("stretch_rows_keep_row_order", [("rows", 0, [20, 50, 120, 300, 0.3])],
     "row 0: must hold x1, y1, x2, y2, score and class"),
    ("stretch_rows_keep_row_order", [("rows", 0, 4, "nan")], f"row 0: {PORTABLE}"),
    ("stretch_rows_keep_row_order", [("rows", 0, 4, float("nan"))], f"row 0: {PORTABLE}"),
    ("stretch_rows_keep_row_order", [("rows", 0, 5, False)], f"row 0: {PORTABLE}"),
    ("stretch_rows_keep_row_order", [("mapping", "scale_y", 0.5)],
     "mapping: a scale times the frame's size is not the content size"),
    ("letterbox_top_padding", [("mapping", "pad_y", 300)], "mapping: padding plus content does not fit the input"),
    ("letterbox_top_padding", [("mode", "stretch")], "mapping: a stretch fills the input and has no padding"),
    ("letterbox_top_padding", [("mapping", "pad_y", -140)],
     "mapping: content sizes must be positive integers and paddings integers of at least zero"),
    ("letterbox_top_padding", [("mapping", "scale_x", 0)], "mapping: scale_x and scale_y must be numbers above zero"),
    ("letterbox_top_padding", [("mapping", "pad", 0)],
     "mapping must hold exactly content_height, content_width, pad_x, pad_y, scale_x, scale_y"),
    ("letterbox_top_padding", [("mode", "crop")],
     "width, height, input_width and input_height must be positive integers, mode stretch or letterbox"),
    ("letterbox_top_padding", [("score_threshold", 0)], "score_threshold must be a number above 0 and at most 1"),
    ("letterbox_top_padding", [("rule", True)], "rule must be 1, 2, 3, 4 or 5"),
    ("letterbox_top_padding", [("rows", [])], "rows must be a non-empty list"),
    ("letterbox_top_padding", [("expected", None)], "expected must be a list"),
)

# Each row: the decode key to overwrite, its value, and the one problem the checks must then name.
BROKEN_SETTINGS = (
    ("layout", "cxcywh_scores_by_anchor", "decode.layout must be xyxy_score_class: the rules and rows are that layout's"),
    ("non_finite", ["nan", "inf", "-inf"], "decode.non_finite must be NaN, Infinity, -Infinity"),
    ("order", "score", "decode.order must be row"),
    ("score_tolerance", 0, "decode.score_tolerance must be a number above 0 and under 0.01"),
    ("coordinate_tolerance", 1, "decode.coordinate_tolerance must be a number above 0 and under 0.5"),
    ("rules", ["1. only one"], "decode.rules must be the five rules, each starting with its number"),
    ("cases", [], "decode.cases must be a non-empty list"),
)


def put(target, *path):
    """Overwrite the value at a path of keys, indexes and slices."""
    *keys, last, value = path
    for key in keys:
        target = target[key]
    target[last] = value


def case(spec, name):
    return next(item for item in spec["decode"]["cases"] if item["name"] == name)


class DecodeContractTests(unittest.TestCase):
    def setUp(self):
        self.spec = checks.load(CONTRACT)

    def test_the_repository_contract_and_models_pass(self):
        self.assertEqual([], checks.contract_problems(self.spec, checks.model_specs(ROOT / "models")))

    def test_each_broken_case_is_named_with_its_one_problem(self):
        for name, changes, problem in BROKEN_CASES:
            spec = copy.deepcopy(self.spec)
            for change in changes:
                put(case(spec, name), *change)
            with self.subTest(case=name, problem=problem):
                self.assertEqual([f"{name}: {problem}"], checks.decode_problems(spec))

    def test_each_broken_setting_is_named(self):
        for key, value, problem in BROKEN_SETTINGS:
            spec = copy.deepcopy(self.spec)
            spec["decode"][key] = value
            with self.subTest(key=key):
                self.assertEqual([problem], checks.decode_problems(spec))

    def test_a_missing_section_or_key_is_named(self):
        keys = "cases, coordinate_tolerance, layout, non_finite, order, rules, score_tolerance"
        for change in (lambda spec: spec.pop("decode"), lambda spec: spec["decode"].pop("order")):
            spec = copy.deepcopy(self.spec)
            change(spec)
            self.assertEqual([f"decode must hold exactly {keys}"], checks.decode_problems(spec))
        case(self.spec, "letterbox_top_padding").pop("rule")
        problems = checks.decode_problems(self.spec)
        self.assertEqual(1, len(problems))
        self.assertIn("a decode case must hold exactly expected, height, input_height", problems[0])

    def test_a_case_without_a_name_is_refused(self):
        for nameless in ("", None):
            spec = copy.deepcopy(self.spec)
            case(spec, "letterbox_top_padding")["name"] = nameless
            with self.subTest(name=nameless):
                self.assertEqual(["a decode case needs a name"], checks.decode_problems(spec))

    def test_every_rule_needs_a_case_and_every_name_is_used_once(self):
        spec = copy.deepcopy(self.spec)
        case(spec, "corners_finite_and_ordered")["rule"] = 5
        self.assertEqual(["rule 3 has no decode case"], checks.decode_problems(spec))
        case(self.spec, "letterbox_top_padding")["name"] = "letterbox_odd_padding"
        self.assertEqual(["case name letterbox_odd_padding is used twice"], checks.decode_problems(self.spec))

    def test_a_bare_nan_is_refused_because_it_is_not_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.json"
            for token in ("NaN", "Infinity", "-Infinity"):
                path.write_text(f'{{"rows": [[0, 0, 1, 1, {token}, 0]]}}', encoding="utf-8")
                with self.subTest(token=token), self.assertRaisesRegex(ValueError, "is not JSON"):
                    checks.load(path)

    def test_a_published_case_may_not_go_or_change_and_a_new_one_is_recorded(self):
        cases = self.spec["decode"]["cases"]
        gone = cases.pop(4)
        missing = "decode case overlapping_boxes_all_stay is missing: a published case stays"
        self.assertEqual([missing], checks.published_problems(self.spec))
        gone["rows"][1][0] = 173
        cases.insert(4, gone)
        changed = (f"decode case overlapping_boxes_all_stay changed (digest {checks.case_digest(gone)}): "
                   "a published case changes only with schema_version")
        self.assertEqual([changed], checks.published_problems(self.spec))
        cases[4] = {**gone, "name": "another_pair"}
        digest = checks.case_digest(cases[4])
        unrecorded = f"decode case another_pair is not recorded: add its digest {digest} to PUBLISHED"
        self.assertEqual([missing, unrecorded], checks.published_problems(self.spec))

    def test_a_digest_ignores_how_the_file_is_laid_out(self):
        first = self.spec["decode"]["cases"][0]
        again = json.loads(json.dumps(first, indent=8, sort_keys=True))
        self.assertEqual(checks.PUBLISHED[first["name"]], checks.case_digest(again))

    def test_a_model_names_only_listed_metadata_values(self):
        models = {"deim": DEIM, "llm": {"kind": "text-generation"}}
        self.assertEqual([], checks.metadata_problems(self.spec, models))
        rotated = copy.deepcopy(DEIM)
        rotated["onnx"]["metadata"]["kataglyphis.output"] = "rotated_boxes"
        del rotated["onnx"]["metadata"]["kataglyphis.preprocess"]
        layouts = "xyxy_score_class, cxcywh_objectness_scores, cxcywh_scores_by_anchor"
        self.assertEqual(["m: kataglyphis.preprocess None is not one of stretch, letterbox",
                          f"m: kataglyphis.output 'rotated_boxes' is not one of {layouts}"],
                         checks.metadata_problems(self.spec, {"m": rotated}))

    def test_the_metadata_keys_and_their_values_are_listed(self):
        listed = self.spec["onnx_metadata"]
        listed["kataglyphis.output"] = ["cxcywh_scores_by_anchor"]
        layout = "onnx_metadata kataglyphis.output must allow xyxy_score_class, the layout the decode cases are for"
        self.assertEqual([layout], checks.metadata_problems(self.spec, {}))
        values = "onnx_metadata kataglyphis.preprocess: the allowed values must be distinct non-empty strings"
        for broken in ([], ["stretch", "stretch"], ["stretch", ""], "stretch"):
            listed["kataglyphis.preprocess"] = broken
            self.assertEqual([values], checks.metadata_problems(self.spec, {}))
        del listed["kataglyphis.preprocess"]
        self.assertEqual(["onnx_metadata must list kataglyphis.preprocess and kataglyphis.output"],
                         checks.metadata_problems(self.spec, {}))

    def test_main_summarises_a_sound_contract_and_fails_a_broken_one(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(0, checks.main(CONTRACT, ROOT / "models"))
        self.assertEqual("inference-v1.json: 9 decode cases, 40 rows, layout xyxy_score_class\n", out.getvalue())
        case(self.spec, "box_cut_at_each_frame_edge")["expected"][2]["box"][2] = 1300
        with tempfile.TemporaryDirectory() as tmp:
            broken = Path(tmp) / "broken.json"
            broken.write_text(json.dumps(self.spec), encoding="utf-8")
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(1, checks.main(broken, ROOT / "models"))
        problem = "box_cut_at_each_frame_edge: expected row 2: rule 5: the box reaches past the frame"
        self.assertEqual(f"broken.json: {problem}\n", err.getvalue())


if __name__ == "__main__":
    unittest.main()
