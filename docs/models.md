# Models: what one looks like, and how one is released

The [README](../README.md) lists the models; this page builds, releases, compares and adds one.
What a consumer does with a release (the lock, the cache, the checks) is the family's contract, in
ANTfrastructure's
[`data-repositories.md`](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md).

## What a model looks like

`models/<name>/model.json` pins the upstream repository and commit, its config, and the checkpoint
with its SHA-256; `models/<name>/LICENSE` is the upstream license. The ONNX:

| Tensor | Name | Shape | Holds |
| --- | --- | --- | --- |
| input | `images` | `[1, 3, 640, 640]` | RGB, 0..1 |
| output | `output0` | `[1, 300, 6]` | x1, y1, x2, y2 in the input's pixels, the score, and the class as a 0..79 COCO index |

Its metadata names the preprocessing (`kataglyphis.preprocess`: `stretch` or `letterbox`), so a
consumer needs no setting.

The shared [inference contract](../contracts/README.md) provides portable frame-to-tensor
cases for Python, Rust and C++ tests, including padded frames and rounded letterbox dimensions.

On the Raspberry Pi 5 (2 threads, ONNX Runtime CPU), measured 2026-10-14 against YOLO26n on 12
arrival pictures of three real cat cameras: about 260 ms a run against 190 ms, and a cat found
more often and more surely (0.42 -> 0.63; a lying cat YOLO26n missed was found at 0.64).

## Building

`scripts/build.sh` runs in the family's Linux image, on its pinned CPython and the torch,
torchvision and chain-built ONNX Runtime of its `/opt/venv`; anywhere else it stops, since ONNX
Runtime would then come from PyPI. `pyproject.toml` declares the export dependencies and `uv.lock`
locks their versions and hashes. Each build creates a fresh overlay venv that sees `/opt/venv`
behind its own packages. The shared overlay helper validates the image versions against the lock,
checks dependency metadata before installing, and installs only missing locked packages. The ORT
census and runtime module paths are checked both before and after installation.

Torch, torchvision, ONNX Runtime, NumPy, Pillow, PyYAML and SciPy come from the family image;
they are excluded from uv's resolver because the chain's Python 3.15 Torch is not a PyPI wheel.
Five additional direct dependencies pin packages shared by this lock and the current image. When
either side changes, the build fails until those versions match; update them with the image and
regenerate `uv.lock`, then run `scripts/tests/check-image-lock.sh` in the family image.

The exporter checks all outputs for finite values, compares every confidence with its own
tolerance, and compares confident classes exactly and boxes to one input pixel. A real
two-cat fixture must produce two cats in both runtimes; a blank frame must produce no
confident detection. `scripts/tests` proves that class, score, shape and NaN mutations fail.

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace -e MODELANT_WORK=/tmp/work \
  ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest bash scripts/build.sh deim_dfine_n_coco dist
```

`scripts/check.sh` is the push check (`check.yml`), in the same image: every model has its spec,
licence and notice, the Python regression tests use the image's pinned CPython, and a fresh overlay
is installed and checked against the image and its ORT wheel store. Initialize
`third_party/ANTfrastructure` before running it in a fresh checkout. It resolves that interpreter
with ANTfrastructure's `ci_image_python_require` and still requires the image's
`PYTHON_MAJOR_MINOR` marker, so a host Python cannot pass. The separate `lint-gates.yml` workflow
checks all tracked shell scripts with ANTfrastructure's pinned ShellCheck and warning ratchet on
every push and pull request. It also fails on duplicated tracked blocks; this repository needs no
duplication exceptions. Before adding Bash or PowerShell behavior, search this repository and
direct submodules for a shared implementation; [AGENTS.md](../AGENTS.md) names the reuse rule.

## Dependency updates

From Linux or WSL, run `bash scripts/renovate-local.sh --apply` from this checkout. The thin
wrapper initializes the recorded hub submodule if absent, then uses this repo's
`.github/renovate.json` and the hub's pinned Renovate driver to update supported dependencies,
including `pyproject.toml`, `uv.lock`, GitHub Actions and direct submodule gitlinks. Report and
`--apply --dry-run` do not initialize the hub; initialize it explicitly or provide an existing
checkout before those read-only calls. A refused update has a nonzero exit status and needs a
deliberate fix.

`model.json` pins the exact upstream commit and checkpoint bytes that produced a tested model.
Moving these pins requires rebuilding, checking the new checkpoint SHA-256 and rerunning the
export and regression suite. A general dependency bump cannot validate a replacement model.

The image runs as uid 1001, which must be able to write `dist/`. Measured 2026-10-20 in `:latest`
(Python 3.14.4, torch 2.14.1+cpu, ONNX Runtime 1.30.0): 58 s, and all five files match
`deim_dfine_n_coco-v1`'s `SHA256SUMS` byte for byte.

## Releasing

```bash
git tag deim_dfine_n_coco-v1 && git push origin deim_dfine_n_coco-v1   # CI builds it in the image and publishes it
gh workflow run release.yml --ref develop -f model=deim_dfine_n_coco   # the same build as an artifact, no release
```

The release carries `<name>.onnx`, `<name>.LICENSE` and `<name>.NOTICE` (ship both with the
model), `<name>.model.json`, the upstream checkpoint it came from (a mirror, so a rebuild does not
depend on Google Drive), `<name>.contract.tar.gz` with the shared test inputs and their
permission notices, and `SHA256SUMS`. The contract archive also carries the project `LICENSE`
and `NOTICE`; the shared package verifier checks all four texts before the archive is published.
The shared mutation workflow checks that omission and bypassing the verifier both fail CI.

## Comparing a candidate

Before a model replaces another, run both on real pictures from the cameras that will use it (they
stay local; never commit them):

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest \
  python3 scripts/compare.py --model old.onnx --model new.onnx --class 15 pictures/*.jpg
```

It prints each model's time per run (2 threads, like the cameras) and what it finds, after each
model's own preprocessing (`kataglyphis.preprocess`; a model without it is letterboxed) at the
input size it was exported for.

`export_deim.py --size 480` (or 320) exports the same checkpoint at a smaller input; the anchors
and position embeddings the checkpoint saved for 640 are rebuilt for it. Measured 2026-10-15 on 11
real arrival pictures of three cameras, on a Pi 5 at the cameras' 0.25 threshold:

| Input | Cat found in | Time a run |
| --- | --- | --- |
| 640 | 6 | 273 ms |
| 480 | 4 | 171 ms |
| 320 | 1 | 97 ms |

480 lost both small cats at a picture's edge, and two others fell from 0.86 and 0.63 to 0.40 and
0.35. The release stays at 640.

## Adding a model

A model belongs here when its code **and** its weights are under a permissive license (Apache-2.0,
MIT, BSD-2-Clause, BSD-3-Clause) and its training data allows it, or under one of the two community
licenses the owner approved on 2026-10-30: `LicenseRef-Llama-Community` and `LicenseRef-Gemma`.
Ultralytics' YOLO models (AGPL-3.0) do not qualify.

A community license adds `license_obligations` to `model.json`: what every release and consumer
must do, such as shipping the agreement and its use policy, or displaying "Built with Llama".
`models/<name>/LICENSE` holds the agreement itself. The release notes still state Apache-2.0 for
every model until `release.yml` reads the license from `model.json`. Multimodal Llama
models exclude users resident in the EU, so none is added. `scripts/license_policy.py` is the rule;
`check.sh` fails a model outside it.

1. `models/<name>/model.json` pins the upstream repository, commit, config and checkpoint
   (SHA-256); `models/<name>/LICENSE` is the upstream license and `models/<name>/NOTICE` names
   its authors and what the export changed.
2. An export script writes the `[1, 300, 6]` layout and names the preprocessing in the metadata
   (`scripts/export_deim.py` is the DEIM one).
3. Compare it (above), then tag `<name>-v1`; the consumers move their lock files to the release's
   SHA-256.
4. Add it to the README's table of models.

## Using one

Pin the release asset's URL and SHA-256 in the consumer and verify it after the download, as the
family's scripts do for every binary they fetch. The lock's format and what every fetcher checks:
[`data-repositories.md` § The lock](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md#the-lock).
