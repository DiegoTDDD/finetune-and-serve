"""Fail if the committed dataset does not match the sha256 values in data/manifest.json."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    bad = [
        name
        for name, digest in manifest["sha256"].items()
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != digest
    ]
    if bad:
        sys.exit(f"dataset files changed without a new manifest: {bad}")
    sizes = ", ".join(f"{k} {v['sentences']}" for k, v in manifest["sets"].items())
    print(f"dataset ok: {sizes}")


if __name__ == "__main__":
    main()
