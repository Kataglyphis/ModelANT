# Changelog

## Unreleased

- The release workflow checks out the ANTfrastructure submodule, which `build.sh` has sourced since
  the shared overlay; a tag push failed before building anything. Its notes print each model's
  license and obligations from `model.json` instead of stating Apache-2.0 for every model.
- Build text-generation models for ONNX Runtime GenAI: `scripts/export_genai.py` fetches the pinned
  Hugging Face files through the hub's lock fetcher, exports them with the image's chain GenAI
  builder, checks `contracts/text-generation-v1.json`, and packs a reproducible `.tar.xz` and an
  architecture-independent `.deb`. `build.sh` dispatches on `model.json`'s `kind`. The first model
  is `smollm2_135m_instruct` (int4, CPU). `onnx-ir` joins the export lock, with `sympy` pinned to
  the image's version.
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
