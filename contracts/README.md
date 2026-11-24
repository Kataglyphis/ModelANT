# Inference contract v1

`inference-v1.json` is the portable reference for RGB/RGBA frames, NCHW float32 input,
stride/offset handling, nearest-neighbour sampling, stretch and letterbox mapping.
Consumers run its cases against their production preprocessing; expected tensors are data,
not calls into a second implementation of their algorithm. Alpha is ignored, padding is
114/255, and integer padding leaves the extra pixel on the right/bottom.

## Decoding the output

`decode` fixes how a detector turns the output's rows into detections. It is for the layout
`xyxy_score_class`: each row is `x1, y1, x2, y2, score, class` in model-input pixels
(`output.columns`). A detector of the family follows five rules:

1. A row is a detection only when its score is a finite number and at least the threshold.
2. Its class must be a number from 0 up to, not including, the number of classes
   (`output.class_range` holds the first and the last class id). A class that is no number,
   negative (also -0.5) or past the range is no detection. The class id is the number cut to an
   integer: 15.9 is class 15.
3. Its four corners must be finite and in order: `x2 >= x1` and `y2 >= y1`. A row whose corners
   are equal, a point, stays.
4. No overlap suppression. An end-to-end model's rows are one per object, so two boxes of one
   class that overlap stay two. What fails when a detector suppresses them anyway: OrchestrANT's
   recipe `cv.contract-corner-rows-suppressed`, AccelerANTgine's `cxx.decode-corner-rows-suppressed`
   (its weekly C++ mutation lane), and in OxidANT, which has no suppression code for this layout to
   switch on, the replayed case `overlapping_boxes_all_stay` itself.
5. Each kept box is mapped back to the frame through the preprocessing mapping,
   `(x - pad_x) / scale_x` and `(y - pad_y) / scale_y`, and then cut to the frame: x into
   `[0, width]`, y into `[0, height]`.

The threshold is each case's `score_threshold`; the cases fix no default. The detectors share
one default when a caller names none: **0.25**, the value measured at the cameras (owner
decision 2026-11-13, hub BACKLOG CON235). ObservANT, OxidANT's `KATAGLYPHIS_SCORE_THRESHOLD`,
AccelerANTgine's `YoloConfig` and OrchestrANT's `yolo-monitor --conf` all start from it.

| Case | Rule | Holds |
| --- | --- | --- |
| `stretch_rows_keep_row_order` | 5 | A 1280x1024 frame stretched to 640x640, scales 0.5 and 0.625; two plain rows, the weaker one first, at threshold 0.25. |
| `score_finite_and_at_threshold` | 1 | At threshold 0.5 the scores NaN, Infinity, -Infinity and 0.49999997 are no detection; 0.5 and 0.75 are. |
| `class_in_range_cut_to_integer` | 2 | Classes 80, -1, -0.5, NaN and Infinity are no detection; 15.9 is class 15, 79.5 class 79 and 0 class 0. |
| `corners_finite_and_ordered` | 3 | A NaN, an Infinity and a -Infinity corner, `x2` before `x1` and `y2` before `y1` are no detection; a point stays. |
| `overlapping_boxes_all_stay` | 4 | Two cats at IoU 8/17 (0.47), a third cat and a couch on the first cat's box: four detections. |
| `box_cut_at_each_frame_edge` | 5 | A box past the left, the top, the right and the bottom edge, and one past all four. |
| `letterbox_top_padding` | 5 | A 1280x720 frame letterboxed with 140 rows above it; the content area maps to the whole frame. |
| `letterbox_left_padding` | 5 | A 720x1280 frame letterboxed with 140 columns left of it. |
| `letterbox_box_in_padding_cut` | 5 | Boxes that reach into the padding above and below the frame are cut at its edges. |

A case gives the frame (`width`, `height`), the input size, `mode` and `mapping` under the names
the preprocessing cases use, so a consumer fills its own mapping type from them. It then feeds
`rows` and `score_threshold` to its production decoder and compares the result with `expected`:

- JSON has no NaN and no infinity. A row spells them as the strings `"NaN"`, `"Infinity"` and
  `"-Infinity"` (`decode.non_finite`), which the float parsers of Rust, C++ and Python read.
- `expected` lists the detections in the order of their rows (`decode.order`); each names its
  `row`, `class`, `score` and its `box` as `x1, y1, x2, y2` in frame pixels. The decoder's list
  must be as long, and entry by entry hold the same class, the score within `score_tolerance`
  and every coordinate within `coordinate_tolerance`, both as absolute differences.
- Every expected coordinate is a whole pixel and every scale a binary fraction, so a decoder
  that reports floats and one that cuts coordinates to integers meet the same numbers.

Expected detections are data worked out by hand. `scripts/decode_contract.py` checks their form
and that none breaks a rule, and never computes a box. It records a digest per published case:
a changed or missing case fails `check.sh`, and a new case is appended with its digest.
`scripts/tests/contract-mutations.json` proves each of these checks can fail.

## Full-size frames by digest

`digest_cases` holds four frames at full size, which the small cases above cannot (hub BACKLOG
CON288 (e)): a 2304x1296 RGB frame stretched and letterboxed to 640x640, a frame 1279 pixels wide
whose rows are padded to 3840 bytes, and an RGBA frame 64 bytes into its buffer. A consumer builds
the frame as `digest_cases.frame` says, byte by byte from x, y and the channel, with 171 in every
padding and offset byte, runs its production preprocessing, and compares the SHA-256 of the NCHW
float32 tensor's little-endian bytes with `expected_sha256`. `samples` name five values, to show
where two tensors part. The comparison is bit for bit: pixel / 255 is a float32 division, as the
three detectors divide today; a float64 division rounded to float32 gives the same bits, and
multiplying by 1/255 does not, in 126 of the 256 values. `test_preprocess_contract.py` rebuilds
each frame and matches its digest through `compare.prepare`.

## Raw heads

`raw_decode` fixes the two raw layouts the family keeps (owner decision D6, 2026-11-21):
`cxcywh_objectness_scores`, rows of cx, cy, w, h, objectness and a score per class, and
`cxcywh_scores_by_anchor`, a `[4 + classes, anchors]` matrix whose columns are the candidates.
Its five rules: a candidate scores its objectness times its best class score, or its best class
score alone, the best over every class column and the first column on a tie; it needs at least
the threshold; its corners come from its centre and size in model-input pixels; the candidates
are taken in descending score, ties to the lower candidate, and each suppresses every later one
of its class whose IoU with it is above 0.45, in model-input pixels before any mapping or cut;
the survivors come in that order, mapped and cut as decode rule 5 says.

| Raw-head case | Raw rule | What it shows |
| --- | --- | --- |
| `objectness_times_the_best_class` | 1 | Scores 0.5, 0.46875 and 0.375 from objectness times class; a tie between two class scores is the first class; 0.1875 is under the threshold. |
| `same_class_overlap_above_0_45_keeps_the_stronger` | 4 | IoU 0.47: the weaker of two boxes of one class goes, a far third stays. |
| `same_class_overlap_at_most_0_45_stays` | 4 | IoU 0.43: both stay. |
| `different_classes_never_suppress` | 4 | IoU 0.67 between two classes: both stay. |
| `equal_scores_keep_the_lower_candidate` | 4 | Three candidates at 0.75: the lower of an overlapping pair stays, and equal scores keep candidate order. |
| `anchor_columns_are_the_candidates` | 4 | The anchor layout: columns are candidates, the best class is the score, and suppression works the same. |
| `suppression_in_model_input_pixels` | 5 | IoU 0.44 in model-input pixels, 0.67 once both boxes are cut at the top of a letterboxed frame: both stay. |

Its expected detections are data too, published with their digests, and `decode_contract.py`
holds each to the five rules: the class and score its candidate gives, the order, and that no
confident candidate is missing unless a kept one of its class ranked before it overlaps it above
0.45, and none is kept that such a one overlaps.

## Metadata keys

`onnx_metadata` lists the two keys an export writes into the ONNX file, with the values a
consumer may meet:

| Key | Values | Says |
| --- | --- | --- |
| `kataglyphis.preprocess` | `stretch`, `letterbox` | How a frame becomes the input tensor: the `mode` of the cases above. |
| `kataglyphis.output` | `xyxy_score_class`, `cxcywh_objectness_scores`, `cxcywh_scores_by_anchor` | The output's layout. |

ModelANT's models name `xyxy_score_class`, the layout of the `output` block and of the decode
cases. The other two names are raw YOLO heads, which AccelerANTgine and OrchestrANT also decode;
their cases are in [Raw heads](#raw-heads). `check.sh` fails a `model.json` whose `onnx.metadata`
lacks a key or names another value.

## Fixture and release

`fixtures/two_cats.jpg` is the existing MIT-licensed OxidANT regression image. Its provenance
and permission notice are in `fixtures/NOTICE` and `fixtures/LICENSE`. `fixtures/two_cats.png` is
its lossless decode, so a golden test hands the model the same pixels in every language rather than
three JPEG decoders' output; `NOTICE` pins its SHA-256. Export validation runs
the real image and a blank frame through both PyTorch and ONNX Runtime, comparing finite
outputs, exact confident classes, confidence tolerances and box tolerances. The positive
image must contain at least two cats; the blank image must contain no confident detection.

The model release carries this contract and the fixtures as `<name>.contract.tar.gz`, covered
by its `SHA256SUMS`. Source-only consumers may pin ModelANT as a test submodule; weights
remain checksum-pinned release assets. No production backend depends on Python validation.

# Text-generation contract v1

`text-generation-v1.json` holds chat cases for GenAI model directories: the messages, a token
budget, a fact the answer must contain and the token ids the release export produced under
greedy decoding on the CPU. The export fails a model without a case; consumers can replay the
cases against their own binding of the same release.
