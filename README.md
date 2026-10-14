# ModelANT

The Kataglyphis family's machine-learning models, kept apart from the code that runs them. Each
model is built here from its pinned upstream source and published as a GitHub release; the code
repositories (OmniAccelerANT, OxidANT, AccelerANTgine) fetch it by URL and check its SHA-256, so no
model file sits in their git history.

Only permissively licensed models live here.

| Model | License | Classes | Params | GFLOPs | COCO AP | Input | Release |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `deim_dfine_n_coco` | Apache-2.0 | COCO 80 | 4 M | 7 | 43.0 | 640x640, plain resize | `deim_dfine_n_coco-v1` |

## What a model looks like

`models/<name>/model.json` pins the upstream repository and commit, its config, and the checkpoint
with its SHA-256; `models/<name>/LICENSE` is the upstream license. The ONNX has one input `images`
`[1, 3, 640, 640]` (RGB, 0..1) and one output `output0` `[1, 300, 6]`: x1, y1, x2, y2 in the
input's pixels, the score, and the class as a 0..79 COCO index. Its metadata names the
preprocessing (`kataglyphis.preprocess`: `stretch` or `letterbox`), so a consumer needs no setting.

On the Raspberry Pi 5 (2 threads, ONNX Runtime CPU), measured 2026-10-14 against YOLO26n on 12
arrival pictures of three real cat cameras: about 260 ms a run against 190 ms, and a cat found more
often and more surely (0.42 -> 0.63, a lying cat missed by YOLO26n found at 0.64).

## Releasing

```bash
bash scripts/build.sh deim_dfine_n_coco dist   # locally: venv, pinned checkout, checkpoint, export, check
git tag deim_dfine_n_coco-v1 && git push origin deim_dfine_n_coco-v1   # CI builds and publishes it
```

The release carries `<name>.onnx`, `<name>.LICENSE` and `<name>.NOTICE` (ship both with the model),
`<name>.model.json`, the upstream checkpoint it came from (a mirror, so a rebuild does not depend
on Google Drive) and `SHA256SUMS`.

## Using one

Pin the release asset's URL and SHA-256 in the consumer and verify it after the download, as the
family's scripts do for every other binary they fetch.
