# Exports a DEIM checkpoint to ONNX as kataglyphis_inference parses it: input `images` [1, 3, S, S]
# (RGB, 0..1, plain resize) and output `output0` [1, 300, 6] = x1, y1, x2, y2, score, class, boxes in the
# S x S input's pixels and classes as 0..79 COCO indices. Its metadata names the preprocessing.
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from inference_contract import validate_outputs
from compare import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deim", required=True, help="the DEIM checkout")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-json", required=True, help="models/<name>/model.json, whose metadata goes into the ONNX")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=640)
    a = p.parse_args()

    sys.path.insert(0, a.deim)
    from engine.core import YAMLConfig

    # The decoder's anchors and the encoder's position embeddings are built for this size.
    cfg = YAMLConfig(a.config, resume=a.checkpoint, eval_spatial_size=[a.size, a.size])
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(a.checkpoint, map_location="cpu")
    state = state["ema"]["module"] if "ema" in state else state["model"]
    # At another size the anchors and masks the checkpoint saved for 640 are rebuilt by the model.
    own = cfg.model.state_dict()
    resized = sorted(k for k, v in state.items() if k in own and own[k].shape != v.shape)
    if any(not k.endswith(("anchors", "valid_mask", "pos_embed")) for k in resized):
        sys.exit(f"checkpoint tensors that do not fit size {a.size}: {resized}")
    cfg.model.load_state_dict({k: v for k, v in state.items() if k not in resized}, strict=not resized)

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            # Deploy mode returns the class index before any COCO category remapping.
            self.post = cfg.postprocessor.deploy()
            self.register_buffer("size", torch.tensor([[a.size, a.size]]))

        def forward(self, images):
            labels, boxes, scores = self.post(self.model(images), self.size)
            return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).float()], dim=-1)

    model = Wrapped().eval()
    torch.manual_seed(0)
    probe = torch.rand(1, 3, a.size, a.size)
    with torch.no_grad():
        expected = model(probe).numpy()
    torch.onnx.export(
        model, (probe,), a.out, input_names=["images"], output_names=["output0"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )

    meta = json.load(open(a.model_json))
    proto = onnx.load(a.out)
    # The concatenation leaves the output's dimensions symbolic; they are fixed, so say so.
    dims = proto.graph.output[0].type.tensor_type.shape.dim
    for dim, value in zip(dims, (1, 300, 6)):
        dim.ClearField("dim_param")
        dim.dim_value = value
    for key, value in meta["onnx"]["metadata"].items():
        proto.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
    proto.metadata_props.append(onnx.StringStringEntryProto(key="kataglyphis.model", value=meta["name"]))
    onnx.checker.check_model(proto)
    onnx.save(proto, a.out)

    print(f"checking with onnxruntime {ort.__version__} ({ort.__file__}) against torch {torch.__version__}")
    session = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    got = session.run(None, {"images": probe.numpy()})[0]
    print(f"random probe: {validate_outputs(expected, got)}")
    validate_real_inputs(model, session, a.size)
    print(f"wrote and validated {a.out}")


def validate_real_inputs(model, session, size):
    """Exercise positive detections and empty-scene behaviour through both runtimes."""
    from PIL import Image

    picture = Path(__file__).resolve().parents[1] / "contracts/fixtures/two_cats.jpg"
    with Image.open(picture) as source:
        image, _ = prepare(source.convert("RGB"), "stretch", size)
        positive = (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    for name, pixels in (("two_cats", positive), ("blank", np.zeros_like(positive))):
        with torch.no_grad():
            expected = model(torch.from_numpy(pixels)).numpy()
        got = session.run(None, {"images": pixels})[0]
        metrics = validate_outputs(expected, got, require_positive=name == "two_cats")
        for backend, result in (("torch", expected), ("onnxruntime", got)):
            if name == "two_cats":
                cats = sum(row[4] >= 0.25 and int(row[5]) == 15 for row in result[0])
                if cats < 2:
                    raise ValueError(f"{backend}: positive fixture found {cats} cats, need at least two")
            elif any(row[4] >= 0.5 for row in result[0]):
                raise ValueError(f"{backend}: blank fixture produced a confident detection")
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
