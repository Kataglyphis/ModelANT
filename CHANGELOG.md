# Changelog

## Unreleased

- `contracts/inference-v1.json` gains two sections (hub BACKLOG CON288 (e)); `schema_version` stays 1, as no published
  case changes. `digest_cases` holds four full-size frames compared by the SHA-256 of their NCHW float32 tensor: a
  2304x1296 frame stretched and letterboxed to 640x640, a 1279-pixel-wide frame with rows padded to 3840 bytes, and an
  RGBA frame 64 bytes into its buffer, each frame built byte by byte from a formula, with five sampled values beside
  the digest. `raw_decode` fixes the raw heads the owner kept (D6), `cxcywh_objectness_scores` and
  `cxcywh_scores_by_anchor`, under five rules (objectness times the best class over every class column, the
  threshold, corners from centre and size, same-class suppression above IoU 0.45 in model-input pixels in descending
  score with ties to the lower candidate, then the mapping and cut), with seven cases, two equal scores among them.
  `decode_contract.py` checks both sections and publishes their cases' digests; `test_preprocess_contract.py`
  rebuilds each frame and matches its digest through `compare.prepare`, and four new tests prove the raw and digest
  checks fail. The detectors replay both sections at their next contract lock.

- `contracts/fixtures/two_cats.png`: the lossless RGB decode of `two_cats.jpg` (480×659, Pillow 11.1.0, PNG level 9,
  SHA-256 `c6cd81c5…` in `fixtures/NOTICE`), so the three detectors' golden tests can hand the model the same pixels
  instead of three JPEG decoders' output (hub BACKLOG CON234, its first point). `large-files.allow` freezes it at
  514 KiB; `test_lossless_fixture.py` holds the decode and the digest. The JPEG stays.
- `contracts/README.md` names, under rule 4, what fails when a detector suppresses overlapping rows anyway:
  OrchestrANT's recipe `cv.contract-corner-rows-suppressed`, AccelerANTgine's weekly C++ recipe
  `cxx.decode-corner-rows-suppressed`, and in OxidANT, which has no suppression code for the layout, the replayed
  case `overlapping_boxes_all_stay` (hub BACKLOG CON234, its second point). No data changed.
- `contracts/README.md` names the detectors' one default score threshold, 0.25 (owner decision
  2026-11-13, hub BACKLOG CON235). The cases keep their own thresholds; no data changed.
- `contracts/inference-v1.json` says how the output's rows are decoded, so the detectors of
  OxidANT, AccelerANTgine and OrchestrANT can replay one reference (hub BACKLOG CON234). `decode`
  holds the owner's five rules for rows of the layout `xyxy_score_class` (a finite score of at
  least the threshold, a class inside `class_range` cut to an integer, finite corners in order,
  no overlap suppression, the box mapped back and cut to the frame) and nine cases: raw rows, a
  threshold, a mapping and the expected detections, worked out by hand. `onnx_metadata` lists
  `kataglyphis.preprocess` and `kataglyphis.output` with their allowed values. No existing key
  or case changed, so `schema_version` stays 1; no model is released. `scripts/decode_contract.py`
  checks the new sections in `check.sh`: their form, every rule on the expected detections, a
  recorded digest per published case, and each `model.json`'s metadata values, with a mutation
  recipe per check.
- `export_deim.py` writes D-FINE's box integral as a matrix product of a column (`scripts/onnx_compat.py`). DirectML computes MatMul by a 1-D vector wrongly: on AMD GPUs `deim_dfine_n_coco-v1` returned no detection on the two-cat photo, with every operator on the GPU. CPU results change by under 0.001 pixels. Released as `deim_dfine_n_coco-v2`; the two-cat photo gives the same five detections on the CPU and on a Radeon RX 9070 XT.
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
