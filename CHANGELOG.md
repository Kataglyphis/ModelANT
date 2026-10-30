# Changelog

## Unreleased

- Accept the Llama and Gemma community licenses beside Apache-2.0, MIT and BSD (owner decision
  2026-10-30, ANTfrastructure `docs/language-ai.md`). `scripts/license_policy.py` is the rule; a
  community license must list its `license_obligations`. The metadata check moved from
  `check.sh`'s heredoc into `scripts/model_metadata.py`, unchanged otherwise, so both are
  unit-tested and mutation-proven.
- Include and verify the project license and notice alongside the fixture notices in contract archives.
- Lock export dependencies with uv and validate their image-provided dependencies before installing a fresh overlay.
- Search this repository and its direct submodules before adding Bash or PowerShell code.
- Fail CI on duplicated tracked blocks through the shared duplication scanner, with no frozen
  exceptions in this repository.
- Reuse ANTfrastructure's Python interpreter probe in the image check and rely on the shared
  lint workflow for tracked shell scripts on every push and pull request.
- Use the shared Windows Git helper so a WSL Renovate run can update a Windows checkout without
  treating its CRLF `.gitmodules` as a local edit.
- Bootstrap a missing pinned ANTfrastructure checkout during a real Renovate apply and update
  direct submodules to their remote default tips; reports and previews do not clone.
- Add the shared Renovate configuration and one-command local dependency update wrapper.
- Add shared CI size, complexity, duplication and Ruff/ty reports, with a manual Python
  refactor preview. Generated local reports stay in ignored `logs/`.
- Reject nonfinite model outputs, confidence/class errors and vacuous positive-fixture checks.
- Validate every export on the existing two-cat regression photo and a blank frame.
- Publish a versioned preprocessing contract, portable tensor cases and fixture provenance
  with each model release. Python comparisons use the same nearest sampling and exact
  rounded letterbox mapping as the native consumers.
