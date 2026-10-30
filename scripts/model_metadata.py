# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""Check every models/<name> directory's files and model.json, one summary line per model."""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from license_policy import license_problems


def model_problems(d: pathlib.Path) -> list[str]:
    """Why one model directory is incomplete or its metadata invalid; empty when it is sound."""
    bad = [f"{d}: no {f}" for f in ("model.json", "LICENSE", "NOTICE") if not (d / f).is_file()]
    if not (d / "model.json").is_file():
        return bad
    spec = json.loads((d / "model.json").read_text())
    if spec["name"] != d.name:
        bad.append(f"{d}: name is {spec['name']}")
    bad += license_problems(d.name, spec)
    if not re.fullmatch(r"[0-9a-f]{40}", spec["upstream"]["commit"]):
        bad.append(f"{d}: upstream commit is not a full sha")
    if not re.fullmatch(r"[0-9a-f]{64}", spec["upstream"]["checkpoint"]["sha256"]):
        bad.append(f"{d}: checkpoint sha256 is not 64 hex digits")
    return bad


def main(models: pathlib.Path) -> int:
    """Print each model's license and source; 1 with every problem on stderr."""
    bad = []
    for d in sorted(p for p in models.iterdir() if p.is_dir()):
        bad += model_problems(d)
        if (d / "model.json").is_file():
            spec = json.loads((d / "model.json").read_text())
            print(f"{d.name}: {spec.get('license')}, {spec['upstream']['repository']}")
    for line in bad:
        print(line, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "models")))
