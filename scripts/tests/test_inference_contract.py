# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_contract import validate_outputs


def output(score=0.9):
    result = [[[0.0] * 6 for _ in range(300)]]
    result[0][0] = [10.0, 20.0, 100.0, 120.0, score, 15.0]
    return result


class OutputContractTests(unittest.TestCase):
    def test_valid_positive_and_small_numeric_noise(self):
        expected = output()
        got = copy.deepcopy(expected)
        got[0][0][0] += 0.5
        got[0][0][4] += 0.0001
        self.assertEqual(validate_outputs(expected, got, require_positive=True)["queries"], 1)

    def test_class_change_by_one_fails(self):
        got = output()
        got[0][0][5] += 1
        with self.assertRaisesRegex(ValueError, "class mismatch"):
            validate_outputs(output(), got)

    def test_confidence_drop_fails(self):
        with self.assertRaisesRegex(ValueError, "confidence mismatch"):
            validate_outputs(output(), output(0.0))

    def test_all_output_values_must_be_finite(self):
        for index in range(6):
            for value in (float("nan"), float("inf"), -float("inf")):
                got = output()
                got[0][299][index] = value
                with self.subTest(index=index, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    validate_outputs(output(), got)

    def test_empty_positive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no confident"):
            validate_outputs(output(0.0), output(0.0), require_positive=True)

    def test_blank_still_checks_low_confidence_scores(self):
        with self.assertRaisesRegex(ValueError, "confidence mismatch"):
            validate_outputs(output(0.01), output(0.04))

    def test_invalid_shape_class_and_confidence_fail(self):
        for column, value in ((5, 1.5), (5, 80), (4, -0.1), (4, 1.1)):
            got = output()
            got[0][0][column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_outputs(output(), got)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_outputs(output(), [[]])

    def test_large_coordinate_error_fails(self):
        got = output()
        got[0][0][2] += 1.01
        with self.assertRaisesRegex(ValueError, "box error"):
            validate_outputs(output(), got)


if __name__ == "__main__":
    unittest.main()
