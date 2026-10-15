# Runs ONNX detectors in the [1, N, 6] layout on the same pictures: time per run and what each finds.
# Each model's preprocessing comes from its kataglyphis.preprocess metadata; a model without it is letterboxed,
# as Ultralytics exports expect. Usage:
#   python scripts/compare.py --model a.onnx --model b.onnx [--class 15] [--threads 2] picture.jpg ...
import argparse
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

NAMES = {0: "person", 14: "bird", 15: "cat", 16: "dog", 56: "chair", 57: "couch", 62: "tv"}


def prepare(img, mode, size=640):
    w, h = img.size
    if mode == "stretch":
        x = img.resize((size, size), Image.BILINEAR)
        return x, lambda a, b, c, d: (a * w / size, b * h / size, c * w / size, d * h / size)
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    x = Image.new("RGB", (size, size), (114, 114, 114))
    x.paste(img.resize((nw, nh), Image.BILINEAR), (dx, dy))
    return x, lambda a, b, c, d: ((a - dx) / r, (b - dy) / r, (c - dx) / r, (d - dy) / r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--class", dest="cls", type=int, help="show only this COCO class (15 = cat)")
    p.add_argument("--score", type=float, default=0.25)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("pictures", nargs="+")
    a = p.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    models = []
    for path in a.model:
        s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        mode = s.get_modelmeta().custom_metadata_map.get("kataglyphis.preprocess", "letterbox")
        # The square input the model was exported for, so 640, 480 and 320 exports compare side by side.
        size = s.get_inputs()[0].shape[2]
        models.append((path.rsplit("/", 1)[-1], s, mode, size))

    total = {name: 0.0 for name, _, _, _ in models}
    for picture in a.pictures:
        img = Image.open(picture).convert("RGB")
        print(f"== {picture.rsplit('/', 1)[-1]} ({img.size[0]}x{img.size[1]})")
        for name, s, mode, size in models:
            x, back = prepare(img, mode, size)
            t = (np.asarray(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
            s.run(None, {"images": t})
            start = time.perf_counter()
            for _ in range(a.runs):
                rows = s.run(None, {"images": t})[0][0]
            ms = (time.perf_counter() - start) / a.runs * 1000
            total[name] += ms
            found = [
                f"{NAMES.get(int(c), int(c))} {sc:.2f} [{', '.join(f'{v:.0f}' for v in back(x1, y1, x2, y2))}]"
                for x1, y1, x2, y2, sc, c in rows
                if sc >= a.score and (a.cls is None or int(c) == a.cls)
            ]
            print(f"  {name:28s} {ms:6.1f} ms  {'; '.join(found) or '-'}")
    for name, ms in total.items():
        print(f"mean {name}: {ms / len(a.pictures):.1f} ms per run")


if __name__ == "__main__":
    main()
