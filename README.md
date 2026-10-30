<div align="center">
  <a href="https://jonasheinle.de">
    <img src="https://raw.githubusercontent.com/Kataglyphis/ModelANT/develop/images/logo.svg" alt="ModelANT logo: an ant in a detection box labelled ant 0.99" width="200" />
  </a>

  <h1>ModelANT</h1>

  <h4>ML models for the Kataglyphis projects, built from pinned upstream sources and published as releases.</h4>
</div>

[![Check](https://github.com/Kataglyphis/ModelANT/actions/workflows/check.yml/badge.svg)](https://github.com/Kataglyphis/ModelANT/actions/workflows/check.yml)
[![Quality gates](https://github.com/Kataglyphis/ModelANT/actions/workflows/lint-gates.yml/badge.svg)](https://github.com/Kataglyphis/ModelANT/actions/workflows/lint-gates.yml)

[**Releases**](https://github.com/Kataglyphis/ModelANT/releases)

The Kataglyphis family's machine-learning models, kept apart from the code that runs them. Each
model is built here from its pinned upstream source and published as a GitHub release; the code
repositories (OmniAccelerANT, OxidANT, AccelerANTgine, OrchestrANT) fetch it by URL and check its
SHA-256, so no model file sits in their git history. Every model's license and its obligations are on record.

## Highlights

- **Licenses on record**: Apache-2.0, MIT or BSD, or the Llama and Gemma licenses with their obligations ([docs/models.md](docs/models.md#adding-a-model)).
- **Built from a pinned upstream**: `model.json` pins the repository, commit, config and checkpoint SHA-256.
- **Built and checked in the family's image**: its pinned CPython, torch and chain-built ONNX Runtime, never PyPI's.
- **One release per model version**, tag `<name>-v<N>`, with the ONNX, its `LICENSE` and `NOTICE`, a license-checked contract archive, and `SHA256SUMS`.
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
docker run --rm -v "$PWD:/workspace" -w /workspace -e MODELANT_WORK=/tmp/work \
  ghcr.io/kataglyphis/kataglyphis_beschleuniger:latest bash scripts/build.sh deim_dfine_n_coco dist
```

It builds in the family's Linux image, on its pinned CPython, torch and chain-built ONNX Runtime,
and needs network ([docs/models.md](docs/models.md#building)). A consumer does not build: it pins the release asset's URL
and SHA-256 in a lock, as
[`data-repositories.md` § The lock](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md#the-lock)
describes.

To refresh this repo's Python, workflow and submodule dependencies from Linux or WSL, run
`bash scripts/renovate-local.sh --apply`; it initializes a missing pinned hub. For a report or
`--apply --dry-run`, first initialize the hub or use an existing checkout. The upstream model
commit and checkpoint hash require separate validation in [docs/models.md](docs/models.md#dependency-updates).
Python export dependencies are declared in `pyproject.toml` and locked in `uv.lock`; the image's
Torch and ONNX Runtime are checked and retained during installation.

## Documentation

| Topic | Read |
| --- | --- |
| Shared Bash and PowerShell reuse rules | [AGENTS.md](AGENTS.md) |
| What a model looks like, and how it ran on the cameras | [docs/models.md](docs/models.md#what-a-model-looks-like) |
| Updating dependencies | [docs/models.md](docs/models.md#dependency-updates) |
| Building a model in the image | [docs/models.md](docs/models.md#building) |
| Releasing a model | [docs/models.md](docs/models.md#releasing) |
| Comparing a candidate, and smaller input sizes | [docs/models.md](docs/models.md#comparing-a-candidate) |
| Shared inference tensors and model validation | [contracts/README.md](contracts/README.md) |
| Adding a model | [docs/models.md](docs/models.md#adding-a-model) |
| Why models live outside the code, and how consumers fetch them | [ANTfrastructure: `data-repositories.md`](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/data-repositories.md) |
| CI quality reports, strict duplication checks, Ruff/ty and refactor previews | [Shared quality tooling](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/code-quality-tooling.md) |

## Contributing and contact

ModelANT is one of the [Kataglyphis repositories](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/repository-family.md),
which share one way to contribute and one contact. Licence: the scripts are Apache-2.0, see
[LICENSE](LICENSE) and [NOTICE](NOTICE); each model keeps its upstream licence, shipped in its
release as `<name>.LICENSE` and `<name>.NOTICE`.
