# ModelANT

Models and reference data are checksum-pinned releases; never commit weights. The scripts
and model exports run in ANTfrastructure's family image, using its chain-built ONNX Runtime.
`bash scripts/check.sh` checks metadata, shell scripts and numerical regression tests.
`bash scripts/renovate-local.sh --apply` updates supported repo dependencies via ANTfrastructure;
model source commits and checkpoint hashes require the rebuild and validation in `docs/models.md`.

`contracts/inference-v1.json` is shared with native consumer tests. Change its version when
changing existing cases or their semantics. New cases may be appended. The exporter must
check a real positive fixture, a blank frame, finite outputs, exact confident classes and
separate confidence/coordinate tolerances. A release includes the deterministic contract
archive and its licences in `SHA256SUMS`.

Keep `docs/models.md`, the README and CHANGELOG in step with behaviour changes. Shared family
rules live in [ANTfrastructure](https://github.com/Kataglyphis/ANTfrastructure/blob/develop/docs/repository-family.md).
