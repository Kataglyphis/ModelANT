# Models: what one looks like, and how one is released

The [README](../README.md) lists the models. This page is how one is built, released, compared
and added. What a consumer does with a release (the lock, the cache, the checks) is the family's
contract, kept in ANTfrastructure's
[`data-repositories.md`](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md).

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

## Comparing a candidate

Before a model replaces another, run both on real pictures from the cameras that will use it
(they stay local; never commit them):

```bash
python scripts/compare.py --model old.onnx --model new.onnx --class 15 pictures/*.jpg
```

It prints each model's time per run (2 threads, like the cameras) and what it finds, after each
model's own preprocessing (`kataglyphis.preprocess`; a model without it is letterboxed) at the
input size it was exported for.

`export_deim.py --size 480` (or 320) exports the same checkpoint at a smaller input; the
anchors and position embeddings the checkpoint saved for 640 are rebuilt for it. Measured
2026-10-15 on 11 real arrival pictures of three cameras, on a Pi 5 at the cameras' 0.25
threshold: 640 found the cat in 6 at 273 ms a run, 480 in 4 at 171 ms (it lost both small cats
at a picture's edge, and two others fell from 0.86 and 0.63 to 0.40 and 0.35), 320 in 1 at
97 ms. The release stays at 640.

## Adding a model

A model belongs here when its code **and** its weights are under a permissive license (Apache-2.0,
MIT, BSD) and its training data allows it. Ultralytics' YOLO models (AGPL-3.0) do not qualify.

1. `models/<name>/model.json` pins the upstream repository, commit, config and checkpoint
   (SHA-256); `models/<name>/LICENSE` is the upstream license and `models/<name>/NOTICE` names
   its authors and what the export changed.
2. An export script that writes the `[1, 300, 6]` layout and names the preprocessing in the
   metadata (`scripts/export_deim.py` is the DEIM one).
3. Compare it (above), then tag `<name>-v1`; the consumers move their lock files to the
   release's SHA-256.
4. Add it to the README's table of models.

## Using one

Pin the release asset's URL and SHA-256 in the consumer and verify it after the download, as the
family's scripts do for every other binary they fetch. The lock's format and what every fetcher
checks: [`data-repositories.md` § The lock](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md#the-lock).
