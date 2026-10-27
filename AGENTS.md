# ModelANT

**Reuse before adding Bash or PowerShell code.** Search this repository and every direct
submodule, including `third_party/ANTfrastructure`, for an existing implementation. Call or
extend its canonical owner, passing project-specific values as arguments. For new behavior
needed by multiple consumers, create or extend one shared owner module and call it here.
Keep unique product logic in this repository.

Models and reference data are checksum-pinned releases; never commit weights. The scripts
and model exports run in ANTfrastructure's family image, using its chain-built ONNX Runtime.
`bash scripts/check.sh` checks metadata and numerical regression tests in the family image.
The shared `lint-gates.yml` workflow checks shell scripts on every push and pull request.
`bash scripts/renovate-local.sh --apply` updates supported repo dependencies via ANTfrastructure;
model source commits and checkpoint hashes require the rebuild and validation in `docs/models.md`.
The export environment uses `pyproject.toml` and `uv.lock`; image packages and matching constraints
must satisfy the shared overlay check before installation, and the ORT census runs on both sides.

`contracts/inference-v1.json` is shared with native consumer tests. Change its version when
changing existing cases or their semantics. New cases may be appended. The exporter must
check a real positive fixture, a blank frame, finite outputs, exact confident classes and
separate confidence/coordinate tolerances. A release includes the deterministic contract
archive and its licences in `SHA256SUMS`.
The contract archive includes the project LICENSE and NOTICE as well as fixture notices; the
shared package verifier checks the completed archive before publication.
The reusable mutation workflow proves both that the project license is packed and that failed
verification cannot publish an invalid replacement.

Keep `docs/models.md`, the README and CHANGELOG in step with behaviour changes. Shared family
rules live in [ANTfrastructure](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/repository-family.md).
