<div align="center">
  <a href="https://jonasheinle.de">
    <img src="https://raw.githubusercontent.com/Kataglyphis/ANTfrastructure/develop/images/logo.png" alt="logo" width="200" />
  </a>

  <h1>ModelANT</h1>

  <h4>Permissively licensed ML models for the Kataglyphis projects, built from pinned upstream sources and published as releases.</h4>
</div>

[![Check](https://github.com/Kataglyphis/ModelANT/actions/workflows/check.yml/badge.svg)](https://github.com/Kataglyphis/ModelANT/actions/workflows/check.yml)

[**Releases**](https://github.com/Kataglyphis/ModelANT/releases)

The Kataglyphis family's machine-learning models, kept apart from the code that runs them. Each
model is built here from its pinned upstream source and published as a GitHub release; the code
repositories (OmniAccelerANT, OxidANT, AccelerANTgine, OrchestrANT) fetch it by URL and check its
SHA-256, so no model file sits in their git history. Only permissively licensed models live here.

## Highlights

- **Permissive only**: code and weights under Apache-2.0, MIT or BSD, and training data that allows it.
- **Built from a pinned upstream**: `model.json` pins the repository, commit, config and checkpoint SHA-256.
- **One release per model version**, tag `<name>-v<N>`, with the ONNX, its `LICENSE` and `NOTICE`, and `SHA256SUMS`.
- **One output layout**: `[1, 300, 6]` boxes, score and COCO class; the ONNX metadata names the preprocessing.
- **Compared on real camera pictures** before a model replaces another (`scripts/compare.py`).

## The models

| Model | License | Classes | Params | GFLOPs | COCO AP | Input | Release |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `deim_dfine_n_coco` | Apache-2.0 | COCO 80 | 4 M | 7 | 43.0 | 640x640, plain resize | `deim_dfine_n_coco-v1` |

## Quick start

```bash
git clone --branch develop https://github.com/Kataglyphis/ModelANT.git
cd ModelANT
bash scripts/build.sh deim_dfine_n_coco dist   # venv, pinned checkout, checkpoint, export, check
```

It needs python3, git and network. A consumer does not build: it pins the release asset's URL
and SHA-256 in a lock, as
[`data-repositories.md` § The lock](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md#the-lock)
describes.

## Documentation

| Topic | Read |
| --- | --- |
| What a model looks like, and how it ran on the cameras | [docs/models.md](docs/models.md#what-a-model-looks-like) |
| Releasing a model | [docs/models.md](docs/models.md#releasing) |
| Comparing a candidate, and smaller input sizes | [docs/models.md](docs/models.md#comparing-a-candidate) |
| Adding a model | [docs/models.md](docs/models.md#adding-a-model) |
| Why models live outside the code, and how consumers fetch them | [ANTfrastructure: `data-repositories.md`](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md) |

## Contributing and contact

ModelANT is one of the [Kataglyphis repositories](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/repository-family.md),
which share one way to contribute and one contact. Licence: the scripts are Apache-2.0, see
[LICENSE](LICENSE) and [NOTICE](NOTICE); each model keeps its upstream licence, shipped in its
release as `<name>.LICENSE` and `<name>.NOTICE`.
