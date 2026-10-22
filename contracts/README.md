# Inference contract v1

`inference-v1.json` is the portable reference for RGB/RGBA frames, NCHW float32 input,
stride/offset handling, nearest-neighbour sampling, stretch and letterbox mapping.
Consumers run its cases against their production preprocessing; expected tensors are data,
not calls into a second implementation of their algorithm. Alpha is ignored, padding is
114/255, and integer padding leaves the extra pixel on the right/bottom.

`fixtures/two_cats.jpg` is the existing MIT-licensed OxidANT regression image. Its provenance
and permission notice are in `fixtures/NOTICE` and `fixtures/LICENSE`. Export validation runs
the real image and a blank frame through both PyTorch and ONNX Runtime, comparing finite
outputs, exact confident classes, confidence tolerances and box tolerances. The positive
image must contain at least two cats; the blank image must contain no confident detection.

The model release carries this contract and the fixtures as `<name>.contract.tar.gz`, covered
by its `SHA256SUMS`. Source-only consumers may pin ModelANT as a test submodule; weights
remain checksum-pinned release assets. No production backend depends on Python validation.
