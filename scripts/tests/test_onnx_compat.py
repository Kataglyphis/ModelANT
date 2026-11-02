import pathlib
import sys
import unittest

import numpy as np
import onnxruntime as ort

# onnx comes with the export overlay, not the image: build.sh runs this suite, check.sh skips it.
try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError as error:
    raise unittest.SkipTest(f"no onnx: {error}") from error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from onnx_compat import expand_vector_matmuls, vector_matmuls  # noqa: E402


def graph_model(project_rank):
    """Softmax(x) @ project, D-FINE's box integral, with a 1-D or a 2-D project."""
    project = np.arange(4, dtype=np.float32).reshape((4,) if project_rank == 1 else (4, 1))
    output_dims = [None] if project_rank == 1 else [None, 1]
    graph = helper.make_graph(
        [
            helper.make_node("Softmax", ["x"], ["soft"], axis=-1),
            helper.make_node("MatMul", ["soft", "project"], ["y"], name="integral/MatMul"),
        ],
        "integral",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_dims)],
        [numpy_helper.from_array(project, "project")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def run(model, x):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {"x": x})[0]


class ExpandVectorMatmuls(unittest.TestCase):
    def test_rewrite_removes_the_1d_form_and_keeps_the_result(self):
        model = graph_model(1)
        x = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
        before = run(model, x)
        self.assertEqual(len(vector_matmuls(model)), 1)

        self.assertEqual(expand_vector_matmuls(model), 1)

        onnx.checker.check_model(model, full_check=True)
        self.assertEqual(vector_matmuls(model), [])
        after = run(model, x)
        self.assertEqual(after.shape, before.shape)
        np.testing.assert_allclose(after, before, atol=1e-6)

    def test_only_the_vector_matmul_changes_among_many_nodes(self):
        model = graph_model(1)
        for index in range(200):
            source = "y" if index == 0 else f"tail{index - 1}"
            model.graph.node.append(helper.make_node("Identity", [source], [f"tail{index}"]))
        model.graph.output[0].name = "tail199"
        x = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
        before = run(model, x)

        self.assertEqual(expand_vector_matmuls(model), 1)

        np.testing.assert_allclose(run(model, x), before, atol=1e-6)
        counts = [node.op_type for node in model.graph.node]
        self.assertEqual(counts.count("Identity"), 200)
        self.assertEqual((counts.count("Unsqueeze"), counts.count("Squeeze")), (1, 1))

    def test_a_matrix_product_is_left_alone(self):
        model = graph_model(2)
        self.assertEqual(expand_vector_matmuls(model), 0)
        self.assertEqual([node.op_type for node in model.graph.node], ["Softmax", "MatMul"])


if __name__ == "__main__":
    unittest.main()
