# Copyright (c) 2026 Kataglyphis
# SPDX-License-Identifier: Apache-2.0
"""Build a text-generation model for ONNX Runtime GenAI from its pinned upstream, check it, and pack it."""
import argparse
import contextlib
import importlib
import json
import lzma
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "text-generation-v1.json"
FETCHER = ROOT / "third_party" / "ANTfrastructure" / "linux" / "scripts" / "lib" / "fetch-data-lock.py"
MAINTAINER = "Kataglyphis <dev@kataglyphis.local>"


def upstream_lock(spec: dict) -> str:
    """The hub fetcher's lock for every pinned upstream file, so the download is checked like any other."""
    up = spec["upstream"]
    files = sorted((f["file"], f["sha256"]) for f in [up["checkpoint"], *up["files"]])
    base = f"{up['repository']}/resolve/{up['commit']}"
    return "".join(f"{digest}  {name}  {base}/{name}\n" for name, digest in files)


def fetch_upstream(spec: dict, work: pathlib.Path) -> pathlib.Path:
    """The upstream files in a fresh directory, each verified by SHA-256."""
    src = work / f"upstream-{spec['name']}"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    lock = work / f"{spec['name']}.upstream.lock"
    lock.write_text(upstream_lock(spec))
    subprocess.run([sys.executable, "-I", str(FETCHER), "--lock", str(lock), "--dest", str(src),
                    "--cache", str(work / "cache")], check=True)
    return src


def build(src: pathlib.Path, out: pathlib.Path, genai: dict, cache: pathlib.Path) -> None:
    """Run the GenAI model builder of the image's chain wheel through its Python API."""
    shutil.rmtree(out, ignore_errors=True)
    # Its CLI cannot start on Python 3.15: argparse rejects the stray % in --extra_options' help (v0.17.0).
    builder = importlib.import_module("onnxruntime_genai.models.builder")
    args = (None, str(src), str(out), genai["precision"], genai["execution_provider"], str(cache))
    builder.create_model(*args, **builder.parse_extra_options(*args, []))


def decode(model_dir: pathlib.Path, messages: list, max_new_tokens: int) -> tuple[list[int], str]:
    """Greedy-decode one chat through the model's own template; the new token ids and their text."""
    og = importlib.import_module("onnxruntime_genai")
    model = og.Model(str(model_dir))
    tok = og.Tokenizer(model)
    prompt = tok.apply_chat_template(json.dumps(messages), add_generation_prompt=True)
    ids = [int(t) for t in tok.encode(prompt)]
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(ids) + max_new_tokens, do_sample=False)
    gen = og.Generator(model, params)
    gen.append_tokens(ids)
    new = []
    while not gen.is_done():
        gen.generate_next_token()
        new.append(int(gen.get_next_tokens()[0]))
    return new, tok.decode(new)


def case_problems(case: dict, ids: list[int], text: str) -> list[str]:
    """Why one decoded case breaks the contract; empty when it holds."""
    problems = []
    if case["must_contain"] not in text:
        problems.append(f"{case['name']}: {text!r} does not contain {case['must_contain']!r}")
    if case["expected_ids"] is None:
        problems.append(f"{case['name']}: no expected_ids recorded; this export produced {json.dumps(ids)}")
    elif case["expected_ids"] != ids:
        problems.append(f"{case['name']}: produced {ids}, the contract records {case['expected_ids']}")
    return problems


def contract_problems(model_dir: pathlib.Path, contract: dict) -> list[str]:
    """Run every contract case that names this model directory; a model with no case fails."""
    cases = [c for c in contract["cases"] if c["model"] == model_dir.name]
    if not cases:
        return [f"{model_dir.name}: no case in {CONTRACT.name}"]
    problems = []
    for case in cases:
        ids, text = decode(model_dir, case["messages"], case["max_new_tokens"])
        print(f"{case['name']}: {text!r} {json.dumps(ids)}")
        problems += case_problems(case, ids, text)
    return problems


def _normalised(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def pack(directory: pathlib.Path, archive: pathlib.Path) -> None:
    """A byte-reproducible tar.xz whose one top directory is `directory`'s name."""
    with lzma.open(archive, "wb", preset=9) as xz, tarfile.open(fileobj=xz, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for path in sorted([directory, *directory.rglob("*")]):
            info = _normalised(tar.gettarinfo(str(path), arcname=str(path.relative_to(directory.parent))))
            with path.open("rb") if path.is_file() else contextlib.nullcontext() as data:
                tar.addfile(info, data)


def deb(spec: dict, model_dir: pathlib.Path, notices: list[pathlib.Path], stage: pathlib.Path,
        out: pathlib.Path) -> pathlib.Path:
    """The architecture-independent model package; its copyright file carries every notice."""
    pkg = spec["package"]
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(model_dir, stage / pkg["install_dir"].lstrip("/"))
    doc = stage / "usr" / "share" / "doc" / pkg["name"]
    doc.mkdir(parents=True)
    (doc / "copyright").write_text("\n".join(p.read_text() for p in notices))
    size_kib = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()) // 1024 + 1
    control = stage / "DEBIAN"
    control.mkdir()
    (control / "control").write_text(
        f"Package: {pkg['name']}\nVersion: {spec['release']}\nSection: misc\nPriority: optional\n"
        f"Architecture: all\nMaintainer: {MAINTAINER}\nInstalled-Size: {size_kib}\n"
        f"Homepage: https://github.com/Kataglyphis/ModelANT\nDescription: {spec['summary']}\n")
    for path in [stage, *stage.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
        os.utime(path, (0, 0))
    target = out / f"{pkg['name']}_{spec['release']}_all.deb"
    env = {**os.environ, "SOURCE_DATE_EPOCH": "0"}
    subprocess.run(["dpkg-deb", "--root-owner-group", "-Zxz", "--build", str(stage), str(target)], check=True, env=env)
    return target


def main() -> int:
    """Fetch, build, check and pack one model; 1 when its contract does not hold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", required=True, type=pathlib.Path)
    parser.add_argument("--work", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    spec = json.loads(args.model_json.read_text())
    model_root = args.model_json.parent
    src = fetch_upstream(spec, args.work)
    model_dir = args.work / spec["genai"]["directory"]
    build(src, model_dir, spec["genai"], args.work / "builder-cache")
    problems = contract_problems(model_dir, json.loads(CONTRACT.read_text()))
    for problem in problems:
        print(f"CONTRACT FAIL: {problem}", file=sys.stderr)
    if problems:
        return 1
    pack(model_dir, args.out / spec["file"])
    deb(spec, model_dir, [model_root / "LICENSE", model_root / "NOTICE"], args.work / "deb", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
